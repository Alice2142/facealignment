// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/
// Configuration:
#include <libconfig.h++>
// Video pipeline utilities:
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <dlib/threads.h>
#include <pthread.h>
#include <list>
// UI:
#include <dlib/gui_widgets.h>
// Processing:
#include <cmath>
#include <dlib/opencv.h>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>

using namespace libconfig;
using namespace std;
using namespace dlib;

class DriverAnalyzer: public multithreaded_object
{
public:
    DriverAnalyzer(
	int videoId_, int apiId_, string cascadeName_, double scaleFactor_, int skipFrames_, double downScale_, 
        int lag_, float varSigmaThresh_, float dampingFactor_,
	frontal_face_detector *detector, 
	shape_predictor *estimator
    ): API_ID(apiId_), CASCADE_NAME(cascadeName_), SCALE_FACTOR(scaleFactor_), SKIP_FRAMES(skipFrames_), DOWN_SCALE(downScale_),
    LAG(lag_), VAR_SIGMA_THRESH(varSigmaThresh_), DAMPING_FACTOR(dampingFactor_)
    {
        // Initialize video stream:
	videoStream = new cv::VideoCapture(videoId_);
        // Initialize face detector & landmarks estimator:
	this->detector = detector;
	this->estimator = estimator;

        if (NULL == videoStream || !videoStream->isOpened())
        {
            cerr << "[VideoProducer]: Unable to connect to video" << videoId_ << endl;
            exit(1);
        }

        // Register 
        register_thread(*this,&DriverAnalyzer::produceVideo);
        register_thread(*this,&DriverAnalyzer::detectFace);
        register_thread(*this,&DriverAnalyzer::displayMetrics);

        // Start all
        start();
    }

    ~DriverAnalyzer()
    {
        // Tell the thread() function to stop.  This will cause should_stop() to 
        // return true so the thread knows what to do.
        stop();

        // Wait for the threads to stop before letting this object destruct itself.
        // Also note, you are *required* to wait for the threads to end before 
        // letting this object destruct itself.
        wait();

	delete videoStream;
    }

private:
    // Concurrent queue:
    template <typename T> class ConcurrentQueue
    { 
        list<T>   m_queue;
        pthread_mutex_t m_mutex;
        pthread_cond_t  m_condv;

    public:
        ConcurrentQueue() {
            pthread_mutex_init(&m_mutex, NULL);
            pthread_cond_init(&m_condv, NULL);
        }
        ~ConcurrentQueue() {
            pthread_mutex_destroy(&m_mutex);
            pthread_cond_destroy(&m_condv);
        }
        void add(T item) {
            pthread_mutex_lock(&m_mutex);
            m_queue.push_back(item);
            pthread_cond_signal(&m_condv);
            pthread_mutex_unlock(&m_mutex);
        }
        T remove() {
            pthread_mutex_lock(&m_mutex);
            while (m_queue.size() == 0) {
                pthread_cond_wait(&m_condv, &m_mutex);
            }
            T item = m_queue.front();
            m_queue.pop_front();
            pthread_mutex_unlock(&m_mutex);
            return item;
        }
        int size() {
            pthread_mutex_lock(&m_mutex);
            int size = m_queue.size();
            pthread_mutex_unlock(&m_mutex);
            return size;
        }
    };    

    // Message for monitor queue:
    struct FacialInfo {
        cv::Mat image;
        std::vector<cv::Point> landmarks;
    
        FacialInfo(
            const cv::Mat image_,
            const std::vector<cv::Point>& landmarks_
        ):image(image_.clone()), landmarks(landmarks_)  {}  
    };    

    // Z-score anomaly detector
    class ZAnomalyDetector {
    public:
        ZAnomalyDetector(
            int lag = 24,
            float varSigmaThresh = 6.18f,
            float dampingFactor = 0.372f
        ) {
            // Config:
            this->lag = lag;
            this->varSigmaThresh = varSigmaThresh;
            this->dampingFactor = dampingFactor;
            // Initialize:
            sumFiltered = 0.0f;
            squareSumFiltered = 0.0f;
            avgFiltered = 0.0f;
            varFiltered = 0.0f;
        }
        bool update(float newVal) {
            bool isAnomaly = false;
    
            // Update classifier:
            if (series.size() < lag) 
            {
                series.push_back(newVal);
                sumFiltered += newVal;
                squareSumFiltered += newVal*newVal;
                // Initialize classifier parameter:
                if (series.size() == lag) 
                {
                    avgFiltered = sumFiltered / lag;
                    varFiltered = (squareSumFiltered - sumFiltered*sumFiltered/lag) / (lag-1);  
                }
            } 
            else 
            {
                float diff = avgFiltered - newVal; //tl
                if (diff > sqrt(varSigmaThresh*varFiltered)) 
                {
                    newVal = dampingFactor*newVal + (1-dampingFactor)*series[lag-1];    
                    isAnomaly = true;
                }
                // Shift:
                for (int i = 0; i < lag-1; ++i) 
                {
                    series[i] = series[i+1];
                }
                sumFiltered += newVal - series[lag-1];//tl
                squareSumFiltered += newVal*newVal - series[lag-1]*series[lag-1];
                avgFiltered = sumFiltered / lag;
                varFiltered = (squareSumFiltered - sumFiltered*sumFiltered/lag) / (lag-1); 
                series[lag-1] = newVal;
            }
        
            return isAnomaly;
        }
    private:
        // Config:
        int lag;
        float varSigmaThresh;
        float dampingFactor;
        // State:
        float sumFiltered;
        float squareSumFiltered;
        float avgFiltered;
        float varFiltered;
        // Series buffer:
        std::vector<float> series;
    };    

    // Video input:
    cv::VideoCapture *videoStream = NULL;
    int SKIP_FRAMES;
    int API_ID; //tl
    string CASCADE_NAME;
    double SCALE_FACTOR;
    double DOWN_SCALE;
    // Models:
    frontal_face_detector *detector = NULL;
    shape_predictor *estimator = NULL;
    // Landmarks:
    const int LANDMARK_START = 36;
    const int LANDMARK_END = 48;
    const int LEFT_EYE_BASE = 36;
    const int RIGHT_EYE_BASE = 42;
    const int NUM_EYE_MARKS = 6; 
    // Eye blink analyzer:
    int LAG;
    float VAR_SIGMA_THRESH, DAMPING_FACTOR;

    // Initialize message queues:
    ConcurrentQueue<cv::Mat> processorQueue;
    ConcurrentQueue<FacialInfo> monitorQueue;
    
    // Video pipeline components:
    void produceVideo()
    {
        // Frame buffer:
	cv::Mat frame;
      
	// Skip frame config:
        int frameIdx = 0;

        while (should_stop() == false)
        {
            videoStream->read(frame);

	    if (!(++frameIdx % SKIP_FRAMES)) {
	    	cv::resize(frame, frame, cv::Size(), 1.0 / DOWN_SCALE, 1.0 / DOWN_SCALE, cv::INTER_AREA);
            frameIdx = 0;
	    	processorQueue.add(frame.clone());
	    }
        }
    }

    void detectFace()
    {
	while (processorQueue.size() != 0)
            processorQueue.remove();
	
	        // Load opencv fd model.
        cv::CascadeClassifier ocvCascade;
        std::vector<cv::Rect> ocvRects;
        if (API_ID == 1)
        {
        	ocvCascade.load(CASCADE_NAME);
        }
        
        int ind = 0;
        clock_t t_start, t_end;
        double time = 0;
        string imNamePath = "../frames/img_";
        string imName;
        while (should_stop() == false)
        {
        	if (ind == LAG * 2)
        	{
        		t_start = clock();
        	}
        	else if ( ind == LAG * 2 + 120)
        	{
        		t_end = clock();
        		cout << "time per frame: " << (double)((t_end - t_start) * 1000 / 120 / CLOCKS_PER_SEC) << endl;
        		cout << "API: " << (API_ID ? "opencv" : "dlib" ) << endl;
        		cout << "down scale: " << DOWN_SCALE << endl;
        	}
        	
	    	// Format image:
	    	cv::Mat frame = processorQueue.remove();
            cv_image<bgr_pixel> image(frame);

        	if (ind >= LAG * 2 && ind < LAG * 2 + 120)
        	{
        		imName = imNamePath + to_string(100 + ind);
        		imName += ".jpg";
        		imwrite(imName, frame);
        	}
        	ind++;
        	
            // Detect faces 
            std::vector<rectangle> faces;
            if (API_ID == 0)
            {
	            clock_t td = clock();
            	faces = (*detector)(image);
	            cout << "Only detection: " << (double) (clock() - td) * 1000/ CLOCKS_PER_SEC << endl;
            }
            else
            {
            	ocvRects.clear();
            	ocvDetect(ocvCascade, frame, ocvRects, SCALE_FACTOR);
            	for (int i = 0; i < ocvRects.size(); i++)
            	{
            		faces.push_back(rectangle(ocvRects[i].x, ocvRects[i].y, ocvRects[i].x + ocvRects[i].width, ocvRects[i].y + ocvRects[i].height));
            	}
            }
            // Find the pose of each face.
            for (unsigned long i = 0; i < faces.size(); ++i) {
                full_object_detection shape = (*estimator)(image, faces[i]);
    		
		std::vector<cv::Point> landmarks;
    		for (int i = LANDMARK_START; i < LANDMARK_END; ++i)
    		{
        	    landmarks.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
    		};
		
                FacialInfo facialInfo(frame, landmarks);
                monitorQueue.add(facialInfo);
            }
        }
    }

    void displayMetrics() {
        // Monitor:
        image_window win;
        // Detectors:
        ZAnomalyDetector eyeCloseDetector(
	    LAG,
	    VAR_SIGMA_THRESH, 
	    DAMPING_FACTOR
	);
	// Series:
        const int SERIES_LEN = 30;
        const int MONITOR_HEIGHT = 36;
    	std::vector<cv::Point> eyeBlinkSeries;
    	// State indicators:
    	const cv::Scalar OPEN(0, 255, 0);
    	const cv::Scalar CLOSED(0, 0, 255);

		//tl VideoWriter
		int codec = CV_FOURCC('M', 'J', 'P', 'G');
		bool isColor;
		cv::Mat resultM;		
		cv::VideoWriter writer;
		
        // Initialize series:
        for (int i = 0; i < SERIES_LEN; ++i) {
            eyeBlinkSeries.push_back(cv::Point(i << 1 + 1, MONITOR_HEIGHT));
        }

        while (should_stop() == false)
        {
	   		FacialInfo facialInfo = monitorQueue.remove();
           
           // Parse image:
           cv_image<bgr_pixel> image(facialInfo.image);
           // Parse landmarks:
	   std::vector<cv::Point> leftEyeMarks(facialInfo.landmarks.begin(), facialInfo.landmarks.begin() + NUM_EYE_MARKS);
	   std::vector<cv::Point> rightEyeMarks(facialInfo.landmarks.begin() + NUM_EYE_MARKS, facialInfo.landmarks.end());

           // Detect eye blink:
	   		float EAR = exactEAR(leftEyeMarks) + exactEAR(rightEyeMarks);
     	    bool isEyeClosed = eyeCloseDetector.update(EAR);
     	    //cout << "EAR: " << EAR << endl;
     	    //cout << "isEyeClosed: " << isEyeClosed << endl;

           // Shift:
           for (int i = 0; i < SERIES_LEN - 1; ++i) {
               eyeBlinkSeries[i].y = eyeBlinkSeries[i + 1].y;
           }
           // Push back:
           eyeBlinkSeries[SERIES_LEN - 1].y = (isEyeClosed ? 1 : MONITOR_HEIGHT);

           // Eye landmarks:
           cv::polylines(facialInfo.image, leftEyeMarks, true, cv::Scalar(255,0,0), 2, 16);    // Left eye
           cv::polylines(facialInfo.image, rightEyeMarks, true, cv::Scalar(255,0,0), 2, 16);   // Right Eye
           // Series:
	   		cv::polylines(facialInfo.image, eyeBlinkSeries, false, (isEyeClosed ? CLOSED : OPEN), 2, 16);
           // Display:
           win.set_image(image);
           //tl save to .avi
           resultM = toMat(image);
           if (!writer.isOpened())
			{
				isColor = resultM.type() == CV_8UC3;
           		writer.open("./result.avi", codec, 25.0, resultM.size(), isColor);
           	}
           	else
           	{
           		writer << resultM;
           	}
        }
    }
 
    // EAR calculation, approximated:
    int approximatedSqrt(int x){
    	int a,b;
    	b     = x;
    	a = x = 0x3f;
    	x     = b/x;
    	a = x = (x+a)>>1;
    	x     = b/x;
    	a = x = (x+a)>>1;
    	x     = b/x;
    	x     = (x+a)>>1;
    	return(x);  
    }
    int approximatedDistance(const cv::Point& p, const cv::Point& q) {
        cv::Point diff = p - q;
        return approximatedSqrt(diff.x*diff.x + diff.y*diff.y);
    }
    float approximatedEAR(const std::vector<cv::Point> &landmarks) {
        int height = (
	    approximatedDistance(landmarks[1], landmarks[5]) + approximatedDistance(landmarks[2], landmarks[4])
        ) >> 1;
        int width = approximatedDistance(landmarks[0], landmarks[3]);
    
        return float(height) / width;   
    }
    // EAR calculation, exact:
    int exactDistance(const cv::Point& p, const cv::Point& q) {
        cv::Point diff = p - q;
        return sqrt(diff.x*diff.x + diff.y*diff.y);
    }
    float exactEAR(const std::vector<cv::Point> &landmarks) {
        double height = (
	    exactDistance(landmarks[1], landmarks[5]) + exactDistance(landmarks[2], landmarks[4])
        ) >> 1;
        double width = exactDistance(landmarks[0], landmarks[3]);
    
        return float(height / width);   
    } 
    void ocvDetect(cv::CascadeClassifier ocvCascade, cv::Mat img, std::vector<cv::Rect> &ocvRects, double ocvScaleFactor)
{
	if (img.channels() >= 3)
	{
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	}
	//cv::equalizeHist(img, img);

	ocvCascade.detectMultiScale(img, ocvRects, ocvScaleFactor, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	return;
}
}; // class DriverAnalyzer

int main()
{
    try
    {
	// Read config:
  	Config cfg;
  	// Read the file. If there is an error, report it and exit.
  	try
  	{
    	    cfg.readFile("application.cfg");
  	}
  	catch(const FileIOException &fioex)
  	{
    	    cerr << "[CfgParser]: I/O error while reading application.cfg" << endl;
            exit(1);
  	}
  	catch(const ParseException &pex)
  	{
    	    cerr << "[CfgParser]: Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << endl;
            exit(1);
  	}
		
	const Setting &config = cfg.getRoot();
        // Video stream config:
	const Setting &video = config["config"]["video"];
	int videoId, skipFrames;
	int apiId; //tl
	double downScale; //tl
        if(
  	    !(
	        video.lookupValue("id", videoId) && 
	        video.lookupValue("api", apiId) && //tl
                video.lookupValue("skipFrames", skipFrames) && 
                video.lookupValue("downScale", downScale)
            )
	)
    	    cerr << "[CfgParser]: Parse video stream config failed." << endl;
        // Blink detector:
	const Setting &blinkDetector = config["config"]["blink"];
	int lag;
	double varSigmaThresh, dampingFactor;
        if(
  	    !(
	        blinkDetector.lookupValue("lag", lag) && 
                blinkDetector.lookupValue("varSigmaThresh", varSigmaThresh) && 
                blinkDetector.lookupValue("dampingFactor", dampingFactor)
            )
	)
    	    cerr << "[CfgParser]: Parse video stream config failed." << endl;
	const Setting &ocvFD = config["config"]["cascade"];
	string cascadeName;
	double scaleFactor;
	if ( !(ocvFD.lookupValue("name", cascadeName) && ocvFD.lookupValue("scaleFactor", scaleFactor)))
	{
		cerr << "[CfgParser]: Parse cascade opencv failed." << endl;
	}
	// Initialize face detector:
        frontal_face_detector detector = get_frontal_face_detector();
        // Initialize facial landmark estimator:
        shape_predictor estimator;
        deserialize("shape_predictor_68_face_landmarks.dat") >> estimator;

	// Start driver analyzer:
	DriverAnalyzer driverAnalyzer(
	    videoId, apiId, cascadeName, scaleFactor, skipFrames, downScale,
	    lag, (float)varSigmaThresh, (float)dampingFactor,
            &detector,
            &estimator
	);
	
	driverAnalyzer.wait();
    }
    catch(serialization_error& e)
    {
        cout << "You need the default face landmarking model file to run this application." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

