//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#include <chrono>
#include <opencv2/opencv.hpp>
// #include "UltraFace.hpp"
#include <iostream>

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"
using namespace nvinfer1;

#define CONVERT_ONNX_TO_TRT
#define ONNX_MODEL_NAME "Mb_Tiny_RFB_FD_train_input_320.onnx"
#define TRT_MODEL_NAME "Mb_Tiny_RFB_FD_train_input_320.trt"

#define INPUT_W 320
#define INPUT_H 240
#define INPUT_C 3
#define OUTPUT_SCORES (4420 * 2)
#define OUTPUT_BOXES  (4420 * 4)
#define INDEX_INPUT  0
#define INDEX_SCORES 1
#define INDEX_BOXES  2
const float PIXEL_MEAN[3] = {0.5f, 0.5f, 0.5f};
const float PIXEL_STD[3] = {0.25f, 0.25f, 0.25f};


// https://devtalk.nvidia.com/default/topic/1049024/what-is-the-defaulat-output-format-of-the-jetson-board-camera/?offset=6#5324356
// sudo apt install v4l-utils
// v4l2-ctl -d /dev/video0 --list-formats-ext
static std::string get_tegra_pipeline(int width, int height, int fps)
{
	return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
	std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
	"/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

static bool onnxToTRTModel(const std::string& modelFile, // name of the onnx model
					unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
					IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	assert(builder != nullptr);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
	if ( !parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
	{
		gLogError << "Failure while parsing ONNX file" << std::endl;
		return false;
	}

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(100 << 20);
	builder->setFp16Mode(false);
	builder->setInt8Mode(false);
	// builder->setAverageFindIterations(4);
	// builder->setMinFindIterations(2) ;

	if (false)
	{
		samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
	}

	samplesCommon::enableDLA(builder, false);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we can destroy the parser
	parser->destroy();

	// serialize the engine, then close everything down
	trtModelStream = engine->serialize();
	engine->destroy();
	network->destroy();
	builder->destroy();

	return true;
}

static void doInference(IExecutionContext& context, float* input, float* scores, float* boxes, int batchSize)
{
	assert(context.getEngine().getNbBindings() == 3);
	void* buffers[3];

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[INDEX_INPUT], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[INDEX_SCORES], batchSize * OUTPUT_SCORES * sizeof(float)));
	CHECK(cudaMalloc(&buffers[INDEX_BOXES], batchSize * OUTPUT_BOXES * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[INDEX_INPUT], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(scores, buffers[INDEX_SCORES], batchSize * OUTPUT_SCORES * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(boxes, buffers[INDEX_BOXES], batchSize * OUTPUT_BOXES * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[INDEX_INPUT]));
	CHECK(cudaFree(buffers[INDEX_SCORES]));
	CHECK(cudaFree(buffers[INDEX_BOXES]));
}


int main(int argc, char** argv)
{
	auto sampleTest = gLogger.defineTest("test", argc, const_cast<const char**>(argv));
	gLogger.reportTestStart(sampleTest);

	/*** Create runtime ***/
	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	// runtime->setDLACore(gArgs.useDLACore);

#ifdef CONVERT_ONNX_TO_TRT
	/* create a TensorRT model from the onnx model */
	IHostMemory* trtModelStream{nullptr};
	if (!onnxToTRTModel(ONNX_MODEL_NAME, 1, trtModelStream)) {
		gLogger.reportFail(sampleTest);
		return -1;
	}
	assert(trtModelStream != nullptr);

	/* save serialized model */
	std::ofstream ofs(TRT_MODEL_NAME, std::ios::out | std::ios::binary);
	ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
	ofs.close();

	// deserialize the engine
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
	trtModelStream->destroy();
#else
	string buffer;
	ifstream stream(TRT_MODEL_NAME, ios::binary);
	if (stream) {
		stream >> noskipws;
		copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
	}
	ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
#endif
	assert(engine != nullptr);

	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	// run inference
	float *input = new float[INPUT_W * INPUT_H * INPUT_C];
	float *scores = new float[OUTPUT_SCORES];
	float *boxes = new float[OUTPUT_BOXES];


	cv::VideoCapture cap;
	cap.open(get_tegra_pipeline(INPUT_W, INPUT_H, 30));
	if (!cap.isOpened()) return -1;
	cv::Mat frame;
	while (cap.read(frame)) {
		#pragma omp parallel for
		for (int i = 0; i < INPUT_W * INPUT_H; i++) {
			for (int c = 0; c < INPUT_C; c++) {
				input[i * INPUT_C + c] = (float(frame.data[i * INPUT_C + c]) / 255.0 - PIXEL_MEAN[c]) / PIXEL_STD[c];
			}
		}
		doInference(*context, input, scores, boxes, 1);

		cv::imshow("UltraFace", frame);
		int key = cv::waitKey(1);
		if (key == 'q') break;
	}

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	delete[] scores;
	delete[] boxes;

	cv::destroyAllWindows();

	return 0;
}
