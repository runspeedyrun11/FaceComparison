/*
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is part of the SeetaFace Identification module, containing codes implementing the
* face identification method described in the following paper:
*
*
*   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
*   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
*   In Frontiers of Computer Science.
*
*
* Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
* Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
*
* The codes are mainly developed by Jie Zhang(a Ph.D supervised by Prof. Shiguang Shan)
*
* As an open-source face recognition engine: you can redistribute SeetaFace source codes
* and/or modify it under the terms of the BSD 2-Clause License.
*
* You should have received a copy of the BSD 2-Clause License along with the software.
* If not, see < https://opensource.org/licenses/BSD-2-Clause>.
*
* Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
*
* Note: the above information must be kept whenever or wherever the codes are used.
*
*/

#include<iostream>
using namespace std;

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>
#include <opencv2\opencv.hpp>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

//#pragma comment( lib, cvLIB("core") )
//#pragma comment( lib, cvLIB("imgproc") )
//#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"

#include "math_functions.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include<opencv2\opencv.hpp>
#include <stdio.h>
#include "facedetect-dll.h"

//#pragma comment(lib,"libfacedetect.lib")
#pragma comment(lib,"libfacedetect-x64.lib")
//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000

using namespace seeta;


#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif


int main(int argc, char* argv[]) {
	// Initialize face detection model
	seeta::FaceDetection detector("C:\\Users\\Gene\\Documents\\SeetaFaceEngine-master\\FaceDetection\\model\\seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector("C:\\Users\\Gene\\Documents\\SeetaFaceEngine-master\\FaceAlignment\\model\\seeta_fa_v1.1.bin");

	// Initialize face Identification model 
	FaceIdentification face_recognizer("C:\\Users\\Gene\\Documents\\SeetaFaceEngine-master\\FaceIdentification\\model\\seeta_fr_v1.0.bin");
	std::string test_dir = DATA_DIR + "test_face_recognizer/";

	//load image
	cv::Mat gallery_img_color = cv::imread("C:/Users/Gene/Documents/Edmond.jpg", 1);
	cv::Mat gallery_img_gray;
	cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

	
//-----------------------------------------------//
	cv::VideoCapture cap(0); // open the default camera

	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	if (!pBuffer)
	{
		fprintf(stderr, "Can not alloc buffer.\n");
		return -1;
	}

	bool doLandmark = true;
	int * pResults = NULL;
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	for (;;)
	{
		cv::Mat frame;
		cap >> frame; // get a new frame from camera
		cv::Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		imshow("edges", frame);

		pResults = facedetect_multiview_reinforce(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
			1.2f, 3, 48, 0, doLandmark);

		printf("%d faces detected.\n", (pResults ? *pResults : 0));
		cv::Mat result_multiview_reinforce = frame.clone();;
		cv::Mat image_roi;
		//print the detection results
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];
			cv::Rect region_of_interest = cv::Rect(x, y, w, h);
			image_roi = frame(region_of_interest);
			imshow("image_roi", image_roi);
			printf("face_rect=[%d, %d, %d, %d], neighbors=%d, angle=%d\n", x, y, w, h, neighbors, angle);
			cv::rectangle(result_multiview_reinforce, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
			if (doLandmark)
			{
				for (int j = 0; j < 68; j++)
					circle(result_multiview_reinforce, cv::Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, cv::Scalar(0, 255, 0));
			}

			//-------------------------------------------------------------------------------------------------//
			cv::Mat probe_img_color = image_roi.clone();
			cv::Mat probe_img_gray;
			cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);
			ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
			gallery_img_data_color.data = gallery_img_color.data;

			ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
			gallery_img_data_gray.data = gallery_img_gray.data;

			ImageData probe_img_data_color(probe_img_color.cols, probe_img_color.rows, probe_img_color.channels());
			probe_img_data_color.data = probe_img_color.data;

			ImageData probe_img_data_gray(probe_img_gray.cols, probe_img_gray.rows, probe_img_gray.channels());
			probe_img_data_gray.data = probe_img_gray.data;

			// Detect faces
			std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
			int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());

			std::vector<seeta::FaceInfo> probe_faces = detector.Detect(probe_img_data_gray);
			int32_t probe_face_num = static_cast<int32_t>(probe_faces.size());

			if (gallery_face_num == 0 || probe_face_num == 0)
			{
				std::cout << "Faces are not detected.";
				return 0;
			}

			// Detect 5 facial landmarks
			seeta::FacialLandmark gallery_points[5];
			point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);

			seeta::FacialLandmark probe_points[5];
			point_detector.PointDetectLandmarks(probe_img_data_gray, probe_faces[0], probe_points);

			for (int i = 0; i<5; i++)
			{
				cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,
					cv::Scalar(0, 255, 0));
				cv::circle(probe_img_color, cv::Point(probe_points[i].x, probe_points[i].y), 2,
					cv::Scalar(0, 255, 0));
			}
			//cv::imwrite("gallery_point_result.jpg", gallery_img_color);
			//cv::imwrite("probe_point_result.jpg", probe_img_color);

			// Extract face identity feature
			float gallery_fea[2048];
			float probe_fea[2048];
			face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
			face_recognizer.ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

			// Caculate similarity of two faces
			float sim = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);
			std::cout << sim << endl;
			//-------------------------------------------//






		}
		//imshow("Results_frontal", result_frontal);
		imshow("Results_multiview", result_multiview_reinforce);
		if (cv::waitKey(30) >= 0) break;

	}







	
	system("pause");
	return 0;
}