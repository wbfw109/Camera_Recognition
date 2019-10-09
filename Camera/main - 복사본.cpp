#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;


/*
https://docs.opencv.org/4.1.0/d3/d63/classcv_1_1Mat.html
#include <opencv2/core/mat.hpp>
	Public Member Functions
		bool 	empty () const
		Mat 	clone () const CV_NODISCARD
		void 	copyTo (OutputArray m) const
		void 	copyTo (OutputArray m, InputArray mask) const
https://docs.opencv.org/4.1.0/d8/dfe/classcv_1_1VideoCapture.html
#include <opencv2/videoio.hpp>

https://docs.opencv.org/4.1.0/da/d35/classcv_1_1Range.html
https://docs.opencv.org/4.1.0/db/d4e/classcv_1_1Point__.html
#include <opencv2/core/types.hpp>
https://docs.opencv.org/4.1.0/d1/dd6/classcv_1_1RNG.html
#include <opencv2/core.hpp>
https://docs.opencv.org/4.1.0/d1/da0/classcv_1_1Scalar__.html
#include <opencv2/core/types.hpp>
	typedef Scalar_<double> Scalar;
https://docs.opencv.org/4.1.0/d7/df6/classcv_1_1BackgroundSubtractor.html
#include <opencv2/video/background_segm.hpp>
https://docs.opencv.org/4.1.0/d7/d19/classcv_1_1AgastFeatureDetector.html
#include <opencv2/features2d.hpp>
*/
/*
https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html
	Mat cv::imread	(	const String & 	filename,
						int 	flags = IMREAD_COLOR
					)
https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html
	int cv::waitKey	(	int 	delay = 0	)
		https://en.wikipedia.org/wiki/ASCII
	void cv::imshow	(	const String & 	winname,
					InputArray 	mat 
					)	
	void cv::namedWindow	(	const String & 	winname,
							int 	flags = WINDOW_AUTOSIZE
							)
https://docs.opencv.org/4.1.0/dc/d84/group__core__basic.html
#include <opencv2/core/matx.hpp>
	typedef Vec<int, 4> cv::Vec4i
	typedef Point2i cv::Point
	typedef Point_<int> cv::Point2i
	typedef Point_<double> Point2d;
#include <opencv2/core/cvstd.hpp>
	typedef std::string cv::String

https://docs.opencv.org/4.1.0/d8/d01/group__imgproc__color__conversions.html
#include <opencv2/imgproc.hpp>
	void cv::cvtColor	(	InputArray 	src,
						OutputArray 	dst,
						int 	code,
						int 	dstCn = 0
						)
	void cv::Canny	(	InputArray 	image,
					OutputArray 	edges,
					double 	threshold1,
					double 	threshold2,
					int 	apertureSize = 3,
					bool 	L2gradient = false
					)
https://docs.opencv.org/4.1.0/d4/d86/group__imgproc__filter.html
★ 
	void cv::bilateralFilter	(	InputArray 	src,
								OutputArray 	dst,
								int 	d,
								double 	sigmaColor,
								double 	sigmaSpace,
								int 	borderType = BORDER_DEFAULT
								)
		Applies the bilateral filter to an image.
		The function applies bilateral filtering to the input image, as described in http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp.
			However, it is very slow compared to most filters.
		Sigma values: For simplicity, you can set the 2 sigma values to be the same. If they are small (< 10), the filter will not have much effect, whereas if they are large (> 150), they will have a very strong effect, making the image look "cartoonish".
		Filter size: Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering.
		This filter does not work inplace.
		//
		Parameters
			src			Source 8-bit or floating-point, 1-channel or 3-channel image.
			dst			Destination image of the same size and type as src .
			d			Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
			sigmaColor	Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
			sigmaSpace	Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
			borderType	border mode used to extrapolate pixels outside of the image, see BorderTypes

https://docs.opencv.org/4.1.0/d3/dc0/group__imgproc__shape.html
#include <opencv2/imgproc.hpp>
	void cv::findContours	(	InputArray 	image,
							OutputArrayOfArrays 	contours,
							OutputArray 	hierarchy,
							int 	mode,
							int 	method,
							Point 	offset = Point()
							)

	void cv::findContours	(	InputArray 	image,
							OutputArrayOfArrays 	contours,
							OutputArray 	hierarchy,
							int 	mode,
							int 	method,
							Point 	offset = Point()
							)
	void cv::approxPolyDP	(	InputArray 	curve,
		OutputArray 	approxCurve,
		double 	epsilon,
		bool 	closed
		)
		// Ramer–Douglas–Peucker algorithm
	double cv::contourArea	(	InputArray 	contour,
		bool 	oriented = false
		)
		// The function computes a contour area. Similarly to moments , the area is computed using the Green formula. 
https://docs.opencv.org/4.1.0/d6/d6e/group__imgproc__draw.htm
#include <opencv2/imgproc.hpp>
	void cv::drawContours	(	InputOutputArray 	image,
			InputArrayOfArrays 	contours,
			int 	contourIdx,
			const Scalar & 	color,
			int 	thickness = 1,
			int 	lineType = LINE_8,
			InputArray 	hierarchy = noArray(),
			int 	maxLevel = INT_MAX,
			Point 	offset = Point()
			)
	void cv::line	(	InputOutputArray 	img,
			Point 	pt1,
			Point 	pt2,
			const Scalar & 	color,
			int 	thickness = 1,
			int 	lineType = LINE_8,
			int 	shift = 0 
			)	
https://docs.opencv.org/4.1.0/dc/d84/group__core__basic.html
#include <opencv2/core/cvstd_wrapper.hpp>
	template<typename _Tp >
	using cv::Ptr = typedef std::shared_ptr<_Tp>

https://docs.opencv.org/4.1.0/de/de1/group__video__motion.html
#include <opencv2/video/background_segm.hpp>
	Ptr<BackgroundSubtractorKNN> cv::createBackgroundSubtractorKNN	(	int 	history = 500,
			double 	dist2Threshold = 400.0,
			bool 	detectShadows = true 
			)		

*/


/* readme ★
	if DEBUGGING_MODE = 0, 
		set "local default value setting", "global value setting"
*/

// global value setting
const int DEBUGGING_MODE = 0;
const int CENTER_CAMERA_X = 320;	// 	AM2111 - Product Resolution (640*480 pixels, VGA)
const int CENTER_CAMERA_Y = 240;
const int MIN_AREA_PIXELS = 20000;
const int MAX_AREA_PIXELS = 25000;		// 150 * 150 정도 되는듯...
const double ERROR_RANGE_LENGTH = 0.02;	// rectangle length error range
const double ERROR_RANGE_LENGTH_POSITIVE = 1 + ERROR_RANGE_LENGTH;
const double ERROR_RANGE_LENGTH_NEGATIVE = 1 - ERROR_RANGE_LENGTH;
const double ERROR_RANGE_COS_POSITIVE = 0.02;	// rectangle angle error range
const double ERROR_RANGE_COS_NEGATIVE = 0 - ERROR_RANGE_COS_POSITIVE;

class Capture
{
	Mat image, gray_image, smooth_image, thresh_image, edge_image, contours_image, approxPoly_image;
	Mat history_image;
	VideoCapture cap;
	vector<vector<Point> > contours, contours_poly;
	vector<Vec4i> contours_hierarchy;
	Point frame_center;			// draw function uses int type
	Point axis[2];
	// primitive_type - local default value setting
	int thresh_type = 0;
	int thresh_value = 120;
	int smooth_diameter = 5;	// changing this, causes lag
	int smooth_sigmaColor = 50;
	int smooth_sigmaSpace = 50;
	int edge_lowThreshold = 50;
	int approxPoly = 4;
	int angle_dx, angle_dy;
	// primitive_type
	double area;
	double distance_between_centers;
	double angle_rotated;
	double aspect_ratio;
	double angle[4];
	double length01, length12;
	// fixed-value
	const int max_filter = 9;
	const int max_smooth_sigma = 200;
	const int max_binary_type = 4;
	const int max_binary_value = 255;
	const int max_lowThreshold = 100;
	const int edge_ratio = 3;
	const int edge_kernel_size = 3;
	const int max_approxPoly = 20;
	const int max_rackbar = 100;
	// name
	const String nimage = "1. Orignal image";
	const String nGray_image = "2. Gray image";
	const String nSmooth_image = "3. Smooth image";
	const String nSmooth_trackbar_Diameter = "Sigma Diameter";
	const String nSmooth_trackbar_sigmaColor = "Sigma Color";
	const String nSmooth_trackbar_sigmaSpace = "Sigma Space";
	const String nThresh_image = "4. Threshold image";
	const String nThresh_trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
	const String nThresh_trackbar_value = "Value";
	const String nEdge_image = "5. Edge image";
	const String nEdge_lowThreshold = "Low Threshold";
	const String nContour_image = "6. Contour image";
	const String nApproxPoly_image = "7. Approximated Polygon image";	// + @ (Retrieved Polygon)
	const String nApproxPoly_epsilon = "Epsilon";
	const String nHistory_image = "-. History image";


static void onChange(int v, void* ptr)
{
	// resolve 'this':
	Capture* that = (Capture*)ptr;
	that->realTrack(v);
}
void realTrack(int v)
{
	if (DEBUGGING_MODE)
	{
		bilateralFilter(gray_image, smooth_image, smooth_diameter, smooth_sigmaColor, smooth_sigmaSpace);
		threshold(smooth_image, thresh_image, thresh_value, max_binary_value, thresh_type);
		Canny(thresh_image, edge_image, edge_lowThreshold, edge_lowThreshold * edge_ratio, edge_kernel_size);
		//
		approxPoly_image = Mat::zeros(contours_image.size(), CV_8UC3);
		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(contours[i], contours_poly[i], approxPoly, true);
		}
	}
}
/*
https://answers.opencv.org/question/69459/find-angle-and-rotation-of-point/
	get cosin(~) value from three point	//  
		from pt0->pt1 and from pt0->pt2
	cos table
		https://onlinemschool.com/math/formula/cosine_table/
	
 */
static double getCos(Point pt0, Point pt1, Point pt2)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}



public:
	// open selected camera using selected API
	Capture(int deviceID = 0)		// 0 = open default camera
		: cap(deviceID + CAP_ANY)	// CAP_ANY 0 = autodetect default API
	{        
		if (!cap.isOpened()) {
			cerr << "ERROR! Unable to open camera\n";
			return;
		}

		namedWindow(nimage);		// int flag : default : WINDOW_AUTOSIZE
		namedWindow(nApproxPoly_image);
		namedWindow(nHistory_image);


		if (DEBUGGING_MODE) {
			namedWindow(nGray_image);
			namedWindow(nSmooth_image);
			namedWindow(nThresh_image);
			namedWindow(nEdge_image);
			namedWindow(nContour_image);

			createTrackbar(nSmooth_trackbar_Diameter,
				nSmooth_image, &smooth_diameter,
				max_filter, onChange, this);
			createTrackbar(nSmooth_trackbar_sigmaColor,
				nSmooth_image, &smooth_sigmaColor,
				max_smooth_sigma, onChange, this);
			createTrackbar(nSmooth_trackbar_sigmaSpace,
				nSmooth_image, &smooth_sigmaSpace,
				max_smooth_sigma, onChange, this);
			//
			createTrackbar(nThresh_trackbar_type,
				nThresh_image, &thresh_type,
				max_binary_type, onChange, this); // Create Trackbar to choose type of Threshold
			createTrackbar(nThresh_trackbar_value,
				nThresh_image, &thresh_value,
				max_binary_value, onChange, this); // Create Trackbar to choose Threshold value
			//
			createTrackbar(nEdge_lowThreshold,
				nEdge_image, &edge_lowThreshold,
				max_lowThreshold, onChange, this);
			//
			createTrackbar(nApproxPoly_epsilon,
				nApproxPoly_image, &approxPoly,
				max_approxPoly, onChange, this);
		}
	}



	bool nextFrame()
	{
		cap.read(image);
		// check if we succeeded
		if (image.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			return false;
		}

		// operation with image				
		cvtColor(image, gray_image, COLOR_BGR2GRAY);
		bilateralFilter(gray_image, smooth_image, smooth_diameter, smooth_sigmaColor, smooth_sigmaSpace);
		threshold(smooth_image, thresh_image, thresh_value, max_binary_value, thresh_type);
		Canny(thresh_image, edge_image, edge_lowThreshold, edge_lowThreshold * edge_ratio, edge_kernel_size);
		findContours(edge_image, contours, contours_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		contours_image = Mat::zeros(edge_image.size(), CV_8UC3);
		approxPoly_image = Mat::zeros(contours_image.size(), CV_8UC3);
		//
		circle(approxPoly_image, Point(CENTER_CAMERA_X, CENTER_CAMERA_Y), 5, Scalar(200, 200, 200));	// center	// 
		//
		for (size_t i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(255, 0, 0);	// blue			
			drawContours(contours_image, contours, (int)i, color, 1, LINE_8, contours_hierarchy, 0);	// default thickness 1   //  Maximal level for drawn contours. If it is 0, only the specified contour is drawn. If it i
			// == https://docs.opencv.org/4.1.0/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
		}		
		contours_poly = vector<vector<Point> >(contours.size());
		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(contours[i], contours_poly[i], approxPoly, true);
			drawContours(approxPoly_image, contours_poly, (int)i, Scalar(0, 255, 0));		// green
			//
			area = contourArea(contours_poly[i]);
			if (contours_poly[i].size() == 4 && area > MIN_AREA_PIXELS && area < MAX_AREA_PIXELS) {		// if rectangle, judge angles ★ this must precede than angle judge for simple operation
				angle[0] = getCos(contours_poly[i][0], contours_poly[i][3], contours_poly[i][1]);
				angle[1] = getCos(contours_poly[i][1], contours_poly[i][0], contours_poly[i][2]);
				angle[2] = getCos(contours_poly[i][2], contours_poly[i][1], contours_poly[i][3]);
				angle[3] = getCos(contours_poly[i][3], contours_poly[i][2], contours_poly[i][0]);
				if (DEBUGGING_MODE) {
					cout << "angle[0 ~ 3]: {" << angle[0] << ", " << angle[1] << ", " << angle[2] << ", " << angle[3] << "}" << endl;
				}

				bool isVertical = true;
				for (size_t i = 0; i < contours_poly[i].size(); i++){
					if (angle[i] > ERROR_RANGE_COS_POSITIVE || angle[i] < ERROR_RANGE_COS_NEGATIVE) {
						isVertical = false;
						break;
					}
				}
				// if angles are equal, judge length
				if (isVertical){			
					length01 = sqrt(pow(contours_poly[i][0].x - contours_poly[i][1].x, 2) + pow(contours_poly[i][0].y - contours_poly[i][1].y, 2));
					length12 = sqrt(pow(contours_poly[i][1].x - contours_poly[i][2].x, 2) + pow(contours_poly[i][1].y - contours_poly[i][2].y, 2));
					
				
					// assume that long line is horizontal line and short line is vertical line	
					// select one of base horizontal lines
					if (length01 >= length12) {							
						angle_dx = contours_poly[i][1].x - contours_poly[i][0].x;
						angle_dy = contours_poly[i][1].y - contours_poly[i][0].y;
						axis[0].x = (contours_poly[i][1].x + contours_poly[i][2].x) / 2;
						axis[0].y = (contours_poly[i][1].y + contours_poly[i][2].y) / 2;
						axis[1].x = (contours_poly[i][3].x + contours_poly[i][0].x) / 2;
						axis[1].y = (contours_poly[i][3].y + contours_poly[i][0].y) / 2;
					}
					else {
						angle_dx = contours_poly[i][1].x - contours_poly[i][2].x;
						angle_dy = contours_poly[i][1].y - contours_poly[i][2].y;
						axis[0].x = (contours_poly[i][0].x + contours_poly[i][1].x) / 2;
						axis[0].y = (contours_poly[i][0].y + contours_poly[i][1].y) / 2;
						axis[1].x = (contours_poly[i][2].x + contours_poly[i][3].x) / 2;
						axis[1].y = (contours_poly[i][2].y + contours_poly[i][3].y) / 2;
					}		
					

					aspect_ratio = length01 / length12;				
					// if lines are euqal, execute operation
					if (aspect_ratio < ERROR_RANGE_LENGTH_POSITIVE && aspect_ratio > ERROR_RANGE_LENGTH_NEGATIVE)
					{
						if (DEBUGGING_MODE) {
							cout << "ratio: " << aspect_ratio << endl;
						}

						
						// get Right triangle - angle (if clockwise rotate is positive)
						angle_rotated = atan2(angle_dy, angle_dx) * 180 / 3.14;


						frame_center.x = frame_center.y = 0.;
						String ss1 = "";
						String ss2 = "";
						for (size_t j = 0; j < contours_poly[i].size(); j++)
						{
							String s = "";
							s.append(to_string(j)).append(". (").append(to_string(contours_poly[i][j].x)).append(", ").append(to_string(contours_poly[i][j].y)).append(")");

							frame_center.x += contours_poly[i][j].x;
							frame_center.y += contours_poly[i][j].y;

							circle(approxPoly_image, contours_poly[i][j], 3, Scalar(255, 255, 0));		// 
							putText(approxPoly_image, s, contours_poly[i][j], FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255)); // white
							//
							circle(image, contours_poly[i][j], 3, Scalar(0, 0, 255));		// red
							putText(image, s, contours_poly[i][j], FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 0)); // white
						}
						frame_center.x /= 4;
						frame_center.y /= 4;
						distance_between_centers = sqrt(pow(CENTER_CAMERA_X - frame_center.x, 2) + pow(CENTER_CAMERA_Y - frame_center.y, 2));

						// Ar: Area, DBC: distance_between_centers, An: rotated angle for object (cos value)
						ss1.append("o : (").append(to_string(frame_center.x)).append(", ").append(to_string(frame_center.y)).
							append("), Ar : ").append(format("%2.f", area / 1000).append("k"));
						ss2.append("DBC: ").append(format("%.2f", distance_between_centers)).
							append(", An: ").append(format("%.2f", angle_rotated));

						history_image = Mat::zeros(contours_image.size(), CV_8UC3);
						putText(history_image, ss1, frame_center, FONT_HERSHEY_PLAIN, 1, Scalar(50, 50, 200));	//
						putText(history_image, ss2, Point(frame_center.x, frame_center.y + 20), FONT_HERSHEY_PLAIN, 1, Scalar(100, 100, 200)); // 
						drawContours(history_image, contours_poly, (int)i, Scalar(255, 0, 0));	// green						
						line(history_image, axis[0], axis[1], Scalar(100, 100, 0));	//
						imshow(nHistory_image, history_image);
						
					}
				}
			}
		}



		imshow(nimage, image);
		if (DEBUGGING_MODE) {
			imshow(nGray_image, gray_image);
			imshow(nSmooth_image, smooth_image);
			imshow(nThresh_image, thresh_image);			
			imshow(nEdge_image, edge_image);
			imshow(nContour_image, contours_image);
		}
		
		imshow(nApproxPoly_image, approxPoly_image);


		if (waitKey(10) == 13)		// enter key
			return false;
		return true;
	}
};






int main(int argc, char** argv) {
	// time
	chrono::duration<double> elapsed;
	chrono::system_clock::time_point start;
	chrono::system_clock::time_point finish;

	
	// advance usage: select any API backend
	

	Capture cap(0);


	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		<< "Press any key to terminate" << endl;
	while (1)
	{
		// time
		start = chrono::high_resolution_clock::now();


		if (!cap.nextFrame())
			break;
		
		// time		
		finish = chrono::high_resolution_clock::now();
		elapsed = finish - start;
		cout << "Elapsed time: " << elapsed.count() << " s\n";
	}
	// the camera will be deinitialized automatically in VideoCapture destructor


	return 0;
}


/*
	C++
		cerr : Standard output stream for errors
		https://docs.microsoft.com/en-us/cpp/standard-library/string?view=vs-2019
			https://docs.microsoft.com/ko-kr/cpp/cpp/string-and-i-o-formatting-modern-cpp?view=vs-2019
			// Boost.Format			
		https://docs.microsoft.com/en-us/cpp/cpp/scope-resolution-operator?view=vs-2019
		https://docs.microsoft.com/en-us/cpp/standard-library/chrono?view=vs-2019
			using high_resolution_clock = system_clock;
			struct system_clock;
		https://docs.microsoft.com/en-us/cpp/cpp/constexpr-cpp?view=vs-2019
		https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=vs-2019
			https://modoocode.com/196
		https://docs.microsoft.com/en-us/cpp/preprocessor/hash-ifdef-and-hash-ifndef-directives-c-cpp?view=vs-2019
		https://docs.microsoft.com/en-us/cpp/preprocessor/pragma-directives-and-the-pragma-keyword?view=vs-2019
		https://docs.microsoft.com/en-us/cpp/standard-library/vector?view=vs-2019
		https://docs.microsoft.com/en-us/cpp/standard-library/overloading-the-input-operator-for-your-own-classes?view=vs-2019
		https://docs.microsoft.com/en-us/cpp/standard-library/overloading-the-output-operator-for-your-own-classes?view=vs-2019
		https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/atan-atanf-atanl-atan2-atan2f-atan2l?view=vs-2019
			atan returns the arctangent of x in the range -π/2 to π/2 radians. 
		//
		For Linux
			#include <unistd.h>
				sleep()

	OpenCV tutorial
		https://docs.opencv.org/4.1.0/d7/d9f/tutorial_linux_install.html
		https://docs.opencv.org/4.1.0/db/d64/tutorial_load_save_image.html
		//
		https://docs.opencv.org/4.1.0/d6/d6d/tutorial_mat_the_basic_image_container.html
		https://docs.opencv.org/4.1.0/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
		https://docs.opencv.org/4.1.0/db/d8e/tutorial_threshold.html
			https://docs.opencv.org/4.1.0/da/d97/tutorial_threshold_inRange.html
		https://docs.opencv.org/4.1.0/df/d0d/tutorial_find_contours.html
		https://docs.opencv.org/4.1.0/da/d5c/tutorial_canny_detector.html
		https://docs.opencv.org/4.1.0/d9/dbc/tutorial_generic_corner_detector.html
			https://docs.opencv.org/4.1.0/d4/d7d/tutorial_harris_detector.html
		https://docs.opencv.org/4.1.0/d7/d1d/tutorial_hull.html
		https://docs.opencv.org/4.1.0/da/d0c/tutorial_bounding_rects_circles.html
		https://docs.opencv.org/4.1.0/de/d62/tutorial_bounding_rotated_ellipses.html
		https://docs.opencv.org/4.1.0/dc/d48/tutorial_point_polygon_test.html
		https://docs.opencv.org/4.1.0/d9/d97/tutorial_table_of_content_features2d.html
			https://docs.opencv.org/4.1.0/dd/d92/tutorial_corner_subpixels.html
			https://docs.opencv.org/4.1.0/d8/dd8/tutorial_good_features_to_track.html
		//
		?
			https://docs.opencv.org/4.1.0/d1/dc5/tutorial_background_subtraction.html
				KNN, MO2 .. 배경 없애는 강도가 너무 강함... 수치 조절 가능?

	+ @
		https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220516822775&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=false&from=postView
*/


/*
특이사항
	O
		이미지 가지고 환경설정할수 있도록 코드 짜기 (DEBUGGIN_MODE)
		#define 을 사용하지 않고 constexpr 을 권장한다.
		vector<vector<Point> >
			contour[i][j] 는 한 선분을 구성하는 Point 중 하나이고
			contour[i].size() 는 한 도형을 구성하는 모든 선분의 수이고
				// ☆ 한 도형에서 하나의 선분은 여러개의 선분으로 구성될 수 있다. (??)
			contour.size() 모든 closed 도형의 수이다.
		꼭 Convex 를 확인할 필요 없는듯. 다른 특성으로 다각형 확인
		(0, 0) ↖ 끝부분

	-
		warning : #pragma once in main file
		


1. 영상으로부터 이미지 획득
2. Image to Gray 로 변경
3. Smoothing (blur)
	https://m.blog.naver.com/PostView.nhn?blogId=ledzefflin&logNo=220503016163&proxyReferer=https%3A%2F%2Fwww.google.com%2F
	cv.bilateralFilter() is highly effective in noise removal while keeping edges sharp.
	?
		borderType
4. Thresholding
	어떤 주어진 임계값(threshold)보다 밝은 픽셀들은 모두 흰색으로, 그렇지 않은 픽셀들은 모두 검은색으로 바꾸는 것
		경계 구분을 위함
5. Canny Edge Detect
	input-image	: 8-bit input image.
	모서리(선분) 찾기
	?
		the number of kernel for Sobel Operator
		low threshold
6. Contours
	윤곽 : 동일한( 색상 강도를 가진 부분의 가장 자리 경계를 연결한 선
	cv::findContours
	cv::drawContours
	?
		Vector<4i>
			하나의 벡터에 int 형식의 4개의 요소가 들어갈수있는 말인가?...
7. Serach Rectangles of Polygons
	다각형 근사
	approxPolyDP()
	contourArea()
		?
			Thus, the returned area and the number of non-zero pixels, if you draw the contour using drawContours or fillPoly , can be different.
			Also, the function will most certainly give a wrong results for contours with self-intersections.	

	X
		 현재 contour로부터 꼭지점을 구하는 함수가 아님
			boundingRect()
			boxPoints()
		코너 검출 알고리즘... 필요 없음.. 잘 맞지도 않는듯.. 수치조정을 잘해야하는것도 문제... 이미 contour 구했으니 거기에서 좌표 도출하는게 맞음.
			Harris corner [1988] ~
			Shi & Tomasi [1994] ~
			SIFT - DoG [2004] ~
			FAST [2006] ~
			AGAST [2010]
			// .. 많은데 못알아들어서.. 최근거는 모르겠고... 그냥 소스 있는 것으로 선택..
				terminate called after throwing an instance of 'cv::Exception'
					what() : OpenCV(4.1.0) / home / pi / opencv / opencv - 4.1.0 / modules / imgproc / src / corner.cpp : 265 : error : (-215:Assertion failed) src.type() == CV_8UC1 || src.type() == CV_32FC1 in function 'cornerEigenValsVecs'
	정사각형의 정의
	세 점으로부터 각도(결과물을 위함), 삼각함수 값(정밀도를 위함)구하기
		atan2
			사분면의 위치, 파라미터의 음수값을 고려한 함수
			atan 으로 하면 0 값이 나올 때도 있다
			// https://zzoyu.tistory.com/73



~.
	
	?
		anonymous parameter? no name parameter... (Callback function)
		time chrono 클래스로부터 fps 추출 방법?
		? Warning 뜨는데 왜 뜨는지 모르겠음...
		HSV (색상을 기준으로 처리) vs BGR (?) ? 무엇을 쓰는 게 좋음
			https://docs.opencv.org/4.1.0/da/d97/tutorial_threshold_inRange.html
		RNG rng(12345);		.. .. ?; 가능한식?... 클래스 생성자 있으면 가능한가.. Modern C++은...
			vector<Rect> boundRect(contours.size());
		vector<vector<Point> > 이런거 일부로 띄우는건가?.. 연산자.. >> 랑 섞일수도 있으니까?..
			contour[i][j].size() 는 한 선분을 구성하는 모든 Point의 개수이고
				// vector 이기 때문에 Point 로부터 방향성을 갖는다.
			contour[i].size() 는 한 도형을 구성하는 모든 선분의 수이고
			contour.size() 모든 closed 도형의 수이다.
		fillPoly() ?
		contours 의 벡터구조를 자세히 몰라서.. 꼭지점 사이의 거리를.. 기초계산으로 코드작성함... 벡터연산..
		String 타입은 escape character를 포함하여도 문자 그대로 출력되버린다.
		★ 확인 필요
			angle_rotated 가 각도가 90도보다 크게, 음수로 안나온다는 조건하에 작성


	to do		
		겹치는 사각형 판별
			https://stackoverflow.com/questions/26583649/opencv-c-rectangle-detection-which-has-irregular-side
			ROI 이미지 따로 만들고 색공간 확인?
		사각형 내부 사각형 판별...
			영상 인식부터 개선해야함...
			https://m.blog.naver.com/samsjang/220534805843
		이미지 특성 매칭?

		비디오에서 객체 추적?

		3차원 고려..
		// undistortPoints
		// https://darkpgmr.tistory.com/32 ...
		// https://www.dino-lite.com/products_detail.php?index_m1_id=9&index_m2_id=11&index_id=1
		? 벡터 연산?..필요?
		matchShape
	


https://docs.opencv.org/4.1.0/d2/dbd/tutorial_distance_transform.html

https://docs.opencv.org/4.1.0/d7/d1d/tutorial_hull.html

★ 	Detect two intersecting rectangles separately in opencv
		https://docs.opencv.org/4.1.0/d7/dd4/classcv_1_1GeneralizedHough.html#gsc.tab=0
	or distanceTransform
	//
	https://docs.opencv.org/4.1.0/de/da9/tutorial_template_matching.html

https://docs.opencv.org/4.1.0/d3/dbe/tutorial_opening_closing_hats.html
	https://www.mathworks.com/help/images/morphological-dilation-and-erosion.html

https://docs.opencv.org/4.1.0/dc/da3/tutorial_copyMakeBorder.html

https://docs.opencv.org/4.1.0/d1/da0/tutorial_remap.html

https://docs.opencv.org/4.1.0/d1/dfd/tutorial_motion_deblur_filter.html
https://docs.opencv.org/4.1.0/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html
https://docs.opencv.org/4.1.0/d2/d0b/tutorial_periodic_noise_removing_filter.html


need?
	https://docs.opencv.org/4.1.0/d0/d49/tutorial_moments.html
	//
	https://docs.opencv.org/4.1.0/d9/db0/tutorial_hough_lines.html
	https://docs.opencv.org/4.1.0/d4/d70/tutorial_hough_circle.html
	https://docs.opencv.org/4.1.0/d4/d1b/tutorial_histogram_equalization.html
	https://docs.opencv.org/4.1.0/d8/dbc/tutorial_histogram_calculation.html
	https://docs.opencv.org/4.1.0/d8/dc8/tutorial_histogram_comparison.html
	https://docs.opencv.org/4.1.0/dd/dd7/tutorial_morph_lines_detection.html
	https://docs.opencv.org/4.1.0/da/d7f/tutorial_back_projection.html	
	https://docs.opencv.org/4.1.0/de/d3c/tutorial_out_of_focus_deblur_filter.html
etc.
	const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";
	src = imread( filename, IMREAD_COLOR );
	imwrite( "../../images/Gray_Image.jpg", gray_image );
	?
		CommandLineParser parser




*/