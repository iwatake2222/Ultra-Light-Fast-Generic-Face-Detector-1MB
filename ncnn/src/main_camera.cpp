//
//  main.cpp
//  UltraFaceTest
//
//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//

#include <chrono>
#include <opencv2/opencv.hpp>
#include "UltraFace.hpp"
#include <iostream>

int main(int argc, char **argv)
{
	std::string bin_path = "ncnn.bin";
	std::string param_path = "ncnn.param";
	cv::namedWindow("UltraFace");
	UltraFace ultraface(bin_path, param_path, 320, 4, 0.7); // config model input

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) return -1;
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

	const auto& t0 = std::chrono::steady_clock::now();
	int cnt = 0;
	double sumInferenceTime = 0;

	cv::Mat frame;
	while (cap.read(frame)) {
		ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

		std::vector<FaceInfo> face_info;
		const auto& t00 = std::chrono::steady_clock::now();
		ultraface.detect(inmat, face_info);
		cnt++;
		sumInferenceTime += ((std::chrono::duration<double>)(std::chrono::steady_clock::now() - t00)).count();

		for (int i = 0; i < face_info.size(); i++) {
			auto face = face_info[i];
			cv::Point pt1(face.x1, face.y1);
			cv::Point pt2(face.x2, face.y2);
			cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
		}

		cv::imshow("UltraFace", frame);
		int key = cv::waitKey(1);
		if (key == 'q') break;
	}

	const auto& t1 = std::chrono::steady_clock::now();
	std::chrono::duration<double> timeSpan = t1 - t0;
	printf("Total processing time = %.3lf [msec]\n", timeSpan.count() * 1000.0 / cnt);
	printf("Inference + Pre/Post Processing time = %.3lf [msec]\n", sumInferenceTime * 1000.0 / cnt);

	cv::destroyAllWindows();
	return 0;
}
