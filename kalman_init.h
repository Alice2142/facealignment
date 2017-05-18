#include <dlib/filtering/kalman_filter.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
using namespace dlib;

#define numStates 4;
#define numMeasurements 2;

class kf_init
{
    private:
		// kalman filter parameters.
		matrix<double, 4, 4> A;

		matrix<double, 2, 4> H;

		matrix<double, 4, 4> Q;
		matrix<double, 2, 2> R = identity_matrix<double>(2);
		matrix<double, 4, 4> P = identity_matrix<double>(4);
		matrix<double, 4, 1> xb;

		kalman_filter<4,  2> *kf;

	public:
	    kf_init(kalman_filter<4,  2> *kfilter, double q, double r)
	    {
    		A = 1.0, 0, 1.0, 0,
			0, 1.0, 0, 1.0,
			0, 0, 1, 0,
			0, 0, 0, 1;
			H = 1, 0, 0, 0,
			0, 1, 0, 0;
			xb = 0, 0, 0.1, 0.1;
	        kf = kfilter;	        
    		Q = identity_matrix<double>(4); 
    		R = identity_matrix<double>(2);
	        Q *= q;
	        R *= r;
		    kf->set_transition_model(A);
		    kf->set_observation_model(H);
		    kf->set_process_noise(Q);
		    kf->set_measurement_noise(R);
		    kf->set_estimation_error_covariance(P);
		    kf->set_state(xb);
		}
		kalman_filter<4, 2>* get_kf()
		{
		    return kf;
		}
};
