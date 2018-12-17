///////////////////////////////////////////////////////////////////////////////////
/// Main.cpp
///
/// Caller Function to invoke ellipse fit
///
/// Author: Sriram Emarose
/// Contact: sriram.emarose@gmail.com
///
///////////////////////////////////////////////////////////////////////////////////

#include "pch.h"
#include <iostream>
#include "Ransac_Ellipse_Fit.h"

int main()
{
	EllipseFinderRansac ellipseFit;
	
	cv::Mat frame_Preprocessed = cv::imread("D:\\ellipse_Preprocessed.png", 0);

	// Get all non zero points in a preprocessed binary image
	std::vector<cv::Point> nonZeroPts;
	cv::findNonZero(frame_Preprocessed, nonZeroPts);

	// Fit Ellipse
	ellipseFit.Process(nonZeroPts, 100, 500, 100);

	if (ellipseFit.bestFitEllipses.size() != 0)
	{
		ellipseFit.Draw(frame_Preprocessed);
		cv::imwrite("Detected_Ellipse.jpg", frame_Preprocessed);
	}
	
}
