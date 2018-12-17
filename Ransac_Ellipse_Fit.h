///////////////////////////////////////////////////////////////////////////////////
/// Ransac_Ellipse_Fit.h
///
/// Fit ellipse in a given set of points using RANSAC
///
//
/// Dependencies: 
/// OpenCV, Eigen
///
/// Author: Sriram Emarose
/// Contact: sriram.emarose@gmail.com
///
///////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iterator>
#include <algorithm>
#include <vector>

#define N_POINTS_ELLIPSE_FIT 31

#define PUSH_TO_THREADS



// Class to implement ellipse fitting using RANSAC method
class EllipseFinderRansac 
{
	cv::Point GetRandomPoint(const std::vector<cv::Point> data);

	std::vector<cv::Point> GetNrandomPoints(const std::vector<cv::Point> data, const int nPoints);

	bool FitEllipse(std::vector<cv::Point> dataPts, const double minRadius, const double maxRadius, double fitError);

public:

	// Stores ellipse parameters
	struct Ellipse
	{
		double majorAxisRadius, minorAxisRadius;

		double rotation;

		cv::Point center;

		Ellipse(cv::Point& c, double& a, double& b, double& alpha)
			: center(c), majorAxisRadius(a), minorAxisRadius(b), rotation(alpha)
		{}
	};
	std::vector<Ellipse> ellipses, bestFitEllipses;
	

	// Default constructor and Destructor
	EllipseFinderRansac();
	~EllipseFinderRansac();

	// Implements RANSAC method to find ellipse in a given frame
	bool Process(const std::vector<cv::Point>& pts, const double minRadius, const double maxRadius, const int nIterations = 5);

	// Draw detected ellipse
	bool Draw(cv::Mat& vis);

};
