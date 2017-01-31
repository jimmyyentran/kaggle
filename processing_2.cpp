#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
using namespace cv;
using namespace std;

cv::Mat resizeKeepAspectRatio(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor)
{
    cv::Mat output;

    double h1 = dstSize.width * (input.rows/(double)input.cols);
    double w2 = dstSize.height * (input.cols/(double)input.rows);
    if( h1 <= dstSize.height) {
        cv::resize( input, output, cv::Size(dstSize.width, h1));
    } else {
        cv::resize( input, output, cv::Size(w2, dstSize.height));
    }

    int top = (dstSize.height-output.rows) / 2;
    int down = (dstSize.height-output.rows+1) / 2;
    int left = (dstSize.width - output.cols) / 2;
    int right = (dstSize.width - output.cols+1) / 2;

    cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor );

    return output;
}


int main(){
    String DIRECTORY = "images/processed_manual/";
    String EXTENSION = "jpg";
    String PROCESSED = "images/processed_2/";
    double SIDE_LENGTH = 350;
    Scalar BLACK = Scalar(0,0,0);
    int IMAGE_COUNT = 1584;
    for (int i = 1; i <= IMAGE_COUNT; ++i) {
        String filename = to_string(i) + "." + EXTENSION;
        String relative_filename = DIRECTORY + filename;
        cout << relative_filename << endl;
        Mat src = imread(relative_filename, CV_LOAD_IMAGE_COLOR);
        Mat output = resizeKeepAspectRatio(src, Size(SIDE_LENGTH, SIDE_LENGTH), BLACK);
        imwrite(PROCESSED + filename, output);
        // imshow("scaled", output);
        // waitKey(0);
    }
    return 0;
}
