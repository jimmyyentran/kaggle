#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
using namespace cv;
using namespace std;

Mat rotate(Mat src, double angle)
{
    // get rotation matrix for rotating the image around its center
    cv::Point2f center(src.cols/2.0, src.rows/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(center,src.size(), angle).boundingRect();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;

    cv::Mat dst;
    cv::warpAffine(src, dst, rot, bbox.size());
    return dst;
}

Mat rescale_bounding_box(String pic, bool display)
{
    Mat src;
    // src = imread("shape.jpg", CV_LOAD_IMAGE_COLOR);
    src = imread(pic, CV_LOAD_IMAGE_COLOR);
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY);
    // threshold(gray, gray,200, 255,THRESH_BINARY_INV); //Threshold the gray
    int largest_area=0;
    int largest_contour_index=0;
    Rect bounding_rect;
    vector<vector<Point> > contours; // Vector for storing contour
    vector<Vec4i> hierarchy;
    findContours( gray, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    // iterate through each contour.
    cout << endl << pic << endl;
    for( int i = 0; i< contours.size(); i++ )
    {
        //  Find the area of contour
        double a=contourArea( contours[i],false); 
        if(a>largest_area){
            largest_area=a;
            // Store the index of largest contour
            largest_contour_index=i;               
            // Find the bounding rectangle for biggest contour
            bounding_rect=boundingRect(contours[i]);
        }
    }
    Scalar color( 255,255,255);  // color of the contour in the leaf
    //Draw the contour and rectangle
    // drawContours( src, contours,largest_contour_index, color, CV_FILLED,8,hierarchy);
    // rectangle(src, bounding_rect,  Scalar(0,255,0),2, 8,0);

    Point tl = bounding_rect.tl();
    Size sz = bounding_rect.size();
    double hw_ratio = double(sz.height) / sz.width;
    cout << "height, width:" << sz.height << ", " << sz.width << endl;
    cout << "h/w ratio:" << hw_ratio << endl;

    // namedWindow( "Display window", CV_WINDOW_AUTOSIZE );

    Mat scaled = src(Rect(tl.x, tl.y, sz.width, sz.height));

    if(hw_ratio < 0.80){
        scaled = rotate(scaled, 90);
        cout << "Rotated" << endl;
    }

    if(display){
        imshow("scaled", scaled);
        waitKey(0);
    }

    // imshow( "Display window", src(Rect(10, 50, 10, 50)) );    

    // imshow( "Display window", src );
    // waitKey(0);                                         
    return scaled;
} 

int main(){
    String DIRECTORY = "images/";
    String EXTENSION = "jpg";
    String PROCESSED = "processed/";
    int IMAGE_COUNT = 1584;
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        String filename = to_string(i) + "." + EXTENSION;
        String relative_filename = DIRECTORY + filename;
        Mat proc = rescale_bounding_box(relative_filename , false);
        cout << DIRECTORY + PROCESSED + filename << endl;
        imwrite(DIRECTORY + PROCESSED + filename, proc);
    }

    return 0;
}
