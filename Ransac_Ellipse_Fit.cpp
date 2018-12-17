///////////////////////////////////////////////////////////////////////////////////
/// Ransac_Ellipse_Fit.cpp
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

#include "pch.h"


#include "Ransac_Ellipse_Fit.h"
#include <opencv2/opencv.hpp>
#define PI 3.142

std::vector<std::vector<cv::Point>> pointsPerIteration;
std::vector<double> fitErrorPerIteration;

std::mutex mLock;


// ############## RANSAC ELLIPSE FIT #############################################

//=============================================================================
/// Default constructor
//=============================================================================
EllipseFinderRansac::EllipseFinderRansac() 
{}

//=============================================================================
/// Default Destructor
//=============================================================================
EllipseFinderRansac::~EllipseFinderRansac()
{}


cv::Point EllipseFinderRansac::GetRandomPoint(const std::vector<cv::Point> pts)
{
	const int idx = rand() % pts.size();
	return pts.at(idx);
}

std::vector<cv::Point> EllipseFinderRansac::GetNrandomPoints(const std::vector<cv::Point> pts, const int nPoints)
{
	std::vector<cv::Point> randomPoints;

	for (int i = 0; i < nPoints; i++)
		randomPoints.push_back(GetRandomPoint(pts));

	return randomPoints;
}


//=============================================================================
/// Implements RANSAC method to detect ellipse in a given image
//=============================================================================
bool EllipseFinderRansac::Process(const std::vector<cv::Point>& dataPoints, const double minRadius, const double maxRadius, const int nIterations)
{
	if (dataPoints.size() < N_POINTS_ELLIPSE_FIT)
	{
		std::cout << "\n Not enough points to fit ellipse. Minimum  "<< N_POINTS_ELLIPSE_FIT << " points required \n";
		return false;
	}

	// Clear previous instances
	ellipses.clear();
	bestFitEllipses.clear();

	std::vector<std::thread> threadPool;

	for (int iter = 0; iter < nIterations; iter++)
	{
		std::vector<cv::Point> oneFitPoints = GetNrandomPoints(dataPoints, N_POINTS_ELLIPSE_FIT);

		double fitError = 0;

#ifdef PUSH_TO_THREADS
		threadPool.push_back(std::thread(&EllipseFinderRansac::FitEllipse, this, oneFitPoints, minRadius, maxRadius, fitError));
#else
		if (!FitEllipse(oneFitPoints, minRadius, maxRadius, fitError))
		{
			std::cout << "\n Error in ellipse fit \n";
			return false;
		}
#endif
	}

#ifdef PUSH_TO_THREADS
	// Maker sure all threads are completed
	try {
		for (int i = 0; i < threadPool.size(); i++)
			if (threadPool[i].joinable()) threadPool[i].join();
	}
	catch (...) {}
#endif


	// No ellipse detected
	if (fitErrorPerIteration.size() == 0)
		return false;

	// Get an ellipse with minimum fit error
	const int minErrorIdx = std::distance(fitErrorPerIteration.begin(), min_element(fitErrorPerIteration.begin(), fitErrorPerIteration.end()));
	const double minError = *min_element(fitErrorPerIteration.begin(), fitErrorPerIteration.end());

#if 1
	// Get One best fit ellipse
	bestFitEllipses.push_back(ellipses[minErrorIdx]);
#else
	// Get a couple of best fit ellipse within a given threshold
	for (uint32_t i = 0; i < ellipses.size(); i++)
	{
		if (fitErrorPerIteration[i] < (double)minError + 2)
			bestFitEllipses.push_back(ellipses[i]);
	}
#endif

	return true;
}

bool EllipseFinderRansac::FitEllipse(std::vector<cv::Point> dataPts, const double minRadius, const double maxRadius, double fitError)
{
	if (dataPts.size() == 0)
		return false;

	Eigen::MatrixXd mA(dataPts.size(), 5);
	Eigen::VectorXd mB(dataPts.size());
	Eigen::VectorXd mC(dataPts.size());

	// Populate the coefficient matrices to solve the least square equation
	for (int i = 0; i < dataPts.size(); i++) {

		mA(i, 0) = dataPts[i].x * dataPts[i].y;
		mA(i, 1) = dataPts[i].y * dataPts[i].y;
		mA(i, 2) = dataPts[i].x;
		mA(i, 3) = dataPts[i].y;
		mA(i, 4) = 1;

		mB(i, 0) = -(dataPts[i].x * dataPts[i].x);

		mC(i, 0) = dataPts[i].x;
	}

	// Solve the least square equation ( A'Ax = A'B)
	Eigen::MatrixXd x = (mA.transpose() * mA).ldlt().solve(mA.transpose() * mB);

	// Get the ellipse coefficients
	double b = x(0);
	double c = x(1);
	double d = x(2);
	double e = x(3);
	double f = x(4);

	// Check for condition satisfying an ellipse 
	double ellipseCheck = (b * b) - (4 * c);

	if (ellipseCheck < 0)
	{
		std::cout << " set of points  converges to an ellipse" << std::endl;
	}
	else
	{
		std::cout << " Points does't satisfy ellipse condition (B^2 - 4AC < 0)" << std::endl;
		return false;
	}

	cv::Mat m0 = cv::Mat(3, 3, CV_64F);
	cv::Mat m1 = cv::Mat(2, 2, CV_64F);

	// Represent general equation of ellipse in matrix form
	m0.at<double>(0, 0) = f;
	m0.at<double>(0, 1) = d / 2;
	m0.at<double>(0, 2) = e / 2;

	m0.at<double>(1, 0) = d / 2;
	m0.at<double>(1, 1) = 1;
	m0.at<double>(1, 2) = b / 2;

	m0.at<double>(2, 0) = e / 2;
	m0.at<double>(2, 1) = b / 2;
	m0.at<double>(2, 2) = c;

	m1.at<double>(0, 0) = 1;
	m1.at<double>(0, 1) = b / 2;
	m1.at<double>(1, 0) = b / 2;
	m1.at<double>(1, 1) = c;

	double majorAxis, minorAxis, h, k, alpha;
	cv::Mat eigen;
	cv::eigen(m1, eigen);
	double lambda1, lambda2;

	lambda1 = (eigen.at<double>(1, 0));
	lambda2 = (eigen.at<double>(0, 0));

	// Get ellipse parameters from obtained coefficients
	// Center
	h = ((b * e) - (2 * c * d)) / ((4 * c) - (b * b));
	k = ((b * d) - (2 * e)) / ((4 * c) - (b * b));

	// Major and Minor axis radius
	majorAxis = sqrt(-(cv::determinant(m0)) / ((cv::determinant(m1))) * lambda1);
	minorAxis = sqrt(-(cv::determinant(m0)) / ((cv::determinant(m1)) * lambda2));

	// Rotation
	alpha = (PI / 2) - atan(((1 - c) / b) / 2);

	// Check for minimum and maximum radius of ellipse to be considered
	if (majorAxis >= minRadius && majorAxis <= maxRadius && minorAxis >= minRadius && majorAxis <= maxRadius)
	{
		cv::Point center = cv::Point(h, k);

		mLock.lock();
		ellipses.push_back(Ellipse(center, majorAxis, minorAxis, alpha));
		mLock.unlock();

		// std::cout << " \n ELLIPSE FOUND DUDE \n";
	}
	else
	{
		// std::cout << " \n OOPS.. No ellipses are detected in the image or within the given radius threshold";
		return false;
	}

	// Calculate fit error
	double errors = 0, dist1 = 0, dist2 = 0, error1 = 0, error2 = 0, dist = 0, err = 0;

	for (int i = 0; i < dataPts.size(); i++)
	{
		double xPt = dataPts[i].x;
		double yPt = dataPts[i].y;

		// Substituting x in general form of ellipse
		double xx = xPt * xPt;
		double mul = sqrt((b * b * xx) + (2 * e * b * xPt) - (4 * c * xx) - (4 * c * d * xPt) + (e * e) - (4 * c * f));
		double yhat = -((e / 2) + ((b * xPt) / 2) + (mul / 2)) / c;
		double yhat1 = -((e / 2) + ((b * xPt) / 2) - (mul / 2)) / c;


		double di1 = (yhat) - dataPts[i].y;
		dist1 = di1 * di1;

		double di2 = yhat1 - dataPts[i].y;
		dist2 = di2 * di2;

		if (dist1 <= 5)
		{
			error1 = error1 + dist1;
		}

		if (dist2 <= 5)
		{
			error2 = error2 + dist2;
		}
	}

	mLock.lock();
	fitErrorPerIteration.push_back(error1 + error2);
	mLock.unlock();

	return true;
}


//=============================================================================
/// Draw detected ellipse
//=============================================================================
bool EllipseFinderRansac::Draw(cv::Mat& vis)
{
	// Using Parametric equation of ellipse
	//x = h + ( a * cos(t))
	//y = k + ( b * sin(t))

	// Check for valid image
	if (vis.empty())
	{
		std::cout << " Input image is empty " << std::endl;
		return 0;
	}

	// Sanity
	if (vis.channels() == 1)
		cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
	
	cv::Point pt;

	// Go through all detected ellipse and draw them
	for (int n = 0; n < bestFitEllipses.size(); n++) {
		for (int i = 0; i < 360; i++) {

			pt.x = bestFitEllipses[n].center.x + (bestFitEllipses[n].majorAxisRadius * cos(i));
			pt.y = bestFitEllipses[n].center.y + (bestFitEllipses[n].minorAxisRadius * sin(i));

			cv::circle(vis, bestFitEllipses[n].center, 1, cv::Scalar(0, 0, 255), -1);
			cv::circle(vis, pt, 1, cv::Scalar(0, 0, 255), 1);
		}
	}
	return 1;
}