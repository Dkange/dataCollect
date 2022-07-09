// Dialog1.cpp: 实现文件
//

#include "pch.h"
#include "finalDataC.h"
#include "Dialog1.h"
#include "afxdialogex.h"

//openpose///////////////////////////////////////////////////////////////
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <Eigen/Eigen>

#include <fstream>
#include <time.h>
// Third-party dependencies
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY

#define LenthCount 20
#define AngelCount 4
#define keyPointsCount 25
#define M_PI 3.14
// Display
DEFINE_bool(no_display, false,
	"Enable to disable the visual display.");

//*********************相机参数*********************//
double fx = 610.4365234375;
double fy = 609.2212524414062;
double cx = 322.4535827636719;
double cy = 238.80897521972656;
Eigen::Vector3d camera_pos(-0.000107598956674, 0.0149245066568, 0.000289224844892);
Eigen::Quaterniond camera_q = Eigen::Quaterniond(0.502717638126, -0.501220281032,
	0.496576036996, -0.499465063011);

//*********************我的函数*********************//

//2021-11-18添加滤波程序
//卡尔曼滤波函数开始
//该代码段是对三个关节角度进行卡尔曼滤波
typedef struct {
	double filterValue;  //k-1时刻的滤波值，即是k-1时刻的值
	double kalmanGain;   //   Kalamn增益
	double A;   // x(n)=A*x(n-1)+u(n),u(n)~N(0,Q)
	double H;   // z(n)=H*x(n)+w(n),w(n)~N(0,R)
	double Q;   //预测过程噪声偏差的方差
	double R;   //测量噪声偏差，(系统搭建好以后，通过测量统计实验获得)
	double P;   //估计误差协方差
}  KalmanInfo;

//初始化应该while循环外完成
void Init_KalmanInfo(KalmanInfo* info, double Q, double R)
{
	info->A = 1;	//标量卡尔曼
	info->H = 1;	//
	info->P = 10;	//后验状态估计值误差的方差的初始值（不要为0问题不大）
	info->Q = Q;    //预测（过程）噪声方差 影响收敛速率，可以根据实际需求给出
	info->R = R;    //测量（观测）噪声方差 可以通过实验手段获得
	info->filterValue = 0;// 测量的初始值
}
double myKalmanFilter(KalmanInfo* kalmanInfo, double lastMeasurement)
{
	//预测下一时刻的值
	double predictValue = kalmanInfo->A * kalmanInfo->filterValue;   //x的先验估计由上一个时间点的后验估计值和输入信息给出，此处需要根据基站高度做一个修改

	//求协方差
	kalmanInfo->P = kalmanInfo->A * kalmanInfo->A * kalmanInfo->P + kalmanInfo->Q;  //计算先验均方差 p(n|n-1)=A^2*p(n-1|n-1)+q
	double preValue = kalmanInfo->filterValue;  //记录上次实际坐标的值

	//计算kalman增益
	kalmanInfo->kalmanGain = kalmanInfo->P * kalmanInfo->H / (kalmanInfo->P * kalmanInfo->H * kalmanInfo->H + kalmanInfo->R);  //Kg(k)= P(k|k-1) H’ / (H P(k|k-1) H’ + R)
	//修正结果，即计算滤波值
	kalmanInfo->filterValue = predictValue + (lastMeasurement - kalmanInfo->H * predictValue) * kalmanInfo->kalmanGain;  //利用残余的信息改善对x(t)的估计，给出后验估计，这个值也就是输出  X(k|k)= X(k|k-1)+Kg(k) (Z(k)-H X(k|k-1))
	//更新后验估计
	kalmanInfo->P = (1 - kalmanInfo->kalmanGain * kalmanInfo->H) * kalmanInfo->P;//计算后验均方差  P[n|n]=(1-K[n]*H)*P[n|n-1]
	return  kalmanInfo->filterValue;
}
//在while循环中直接使用该函数进行角度滤波
void KalmanFs(float* lastMeasurement, KalmanInfo* info)
{
	//float anglesLaters[3] = { 0 };
	for (int i = 0; i < 3; i++)
		lastMeasurement[i] = myKalmanFilter(&info[i], lastMeasurement[i]);
	//return anglesLaters;
}
//卡尔曼滤波函数结束


//像素坐标提取函数
int** getPixel(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
	//该函数应该返回一个25*3大小的数组，该数组中3各元素为一组，三个元素分别为x y score
	//datumsPtr中拿到的数据实际上是double类型的小数，但是像素坐标实际应该是整分的，所以在这里四舍五入取整，防止后面计算出错
	int** pixelCoordinates = NULL;
	//动态分配二维数组
	pixelCoordinates = (int**)malloc(keyPointsCount * sizeof(int*));
	if (NULL == pixelCoordinates)
		std::cout << "pixelCoordinates[25]申请内存失败\n";
	for (size_t i = 0; i < keyPointsCount; i++)
	{
		pixelCoordinates[i] = (int*)malloc(3 * sizeof(int));
		if (NULL == pixelCoordinates)
			std::cout << "pixelCoordinates[25][3]申请内存失败\n";
	}

	//这里判断一下datumsPtr中是否有识别到的人体数据，如果内有下面赋值的时候全部赋值为0
	if (datumsPtr->at(0)->poseKeypoints.getConstPtr() == NULL)
	{
		for (size_t i = 0; i < keyPointsCount; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				pixelCoordinates[i][j] = 0;
			}
		}
	}
	else
	{
		for (size_t i = 0; i < keyPointsCount; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				pixelCoordinates[i][j] = round(*(datumsPtr->at(0)->poseKeypoints.getConstPtr() + i * 3 + j));//????
			}
		}
	}
	return pixelCoordinates;
}

//获取深度信息
//参数分别为像素坐标数组和深度frame
float* getDepth(int** pixel, rs2::depth_frame depth_frame)
{
	float* depth;
	depth = (float*)malloc(keyPointsCount * sizeof(float));
	for (size_t i = 0; i < keyPointsCount; i++)
	{
		if (pixel[i][0] == 0)
			depth[i] = 0;
		else
			depth[i] = depth_frame.get_distance(pixel[i][0], pixel[i][1]);

		//调试打印深度信息
		//std::cout << depth[i] << std::endl;
	}
	return depth;
}
//坐标数据写入文件
void writeCoordinate2txt(Eigen::Vector3d* pt_world_arr)
{
	std::string pathName = "F:\\Yan_2\\robotContest\\finaldatacollect\\finalDataC\\KeyPos\\";
	//获取当前时间用于文件命名
	time_t curTime = time(0);
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%Y_%m_%d", localtime(&curTime));
	//strftime(tmp, sizeof(tmp), "%Y_%m_%d_%H_%M_%S", localtime(&curTime));

	pathName.append(tmp);
	pathName.append("_Coordinate.txt");
	//std::cout << pathName << std::endl;
	std::ofstream coordinate;
	coordinate.open(pathName, std::ios::app);	//这里的5表示追加写入，而不是擦除写入
	if (!coordinate.is_open())
		std::cout << "coordinate文件创建\\打开失败，无法进行写入\n";
	//std::ofstream out("写入.txt");
	coordinate << "new frame: \n";
	for (size_t i = 0; i < keyPointsCount; i++)
	{
		coordinate << pt_world_arr[i][0] << " ";
		coordinate << pt_world_arr[i][1] << " ";
		coordinate << pt_world_arr[i][2] << "\n";
	}
	coordinate.close();
}
//坐标重建函数，参数：跟踪目标框的像素位置x y 深度
void loc3d(int** pixel, float* depth, Eigen::Vector3d* pt_world_arr)
{
	Eigen::Matrix3d camera_r = camera_q.toRotationMatrix();
	Eigen::Vector3d pt_cur, pt_world;
	//Eigen::Vector3d pt_world_arr[keyPointsCount];
	for (size_t i = 0; i < keyPointsCount; i++)
	{
		if (depth[i] == 0)
		{
			pt_cur(0) = 0;
			pt_cur(1) = 0;
			pt_cur(2) = 0;
			pt_world = camera_r * pt_cur + camera_pos;	//因为camera_pos有数值，所以当像素值为0的时候pt_world本应该返回000表示无效数据
														//但是还是有较小的数值显示
			pt_world_arr[i] = pt_world;
		}
		else
		{
			pt_cur(0) = (pixel[i][0] - cx) * depth[i] / fx;
			pt_cur(1) = (pixel[i][1] - cy) * depth[i] / fy;
			pt_cur(2) = depth[i];
			pt_world = camera_r * pt_cur + camera_pos;
			pt_world_arr[i] = pt_world;
		}
		//打印调试信息
		//std::cout << "pt_world(0): " << pt_world_arr[i](0) << "," << pt_world_arr[i](1) << "," << pt_world_arr[i](2) << std::endl;

	}
	//return pt_world_arr;
}

//肢体长度计算和关节角度
void calculateDisAngle(Eigen::Vector3d* pt_world_arr, float* len, float* angle)
{
	//分析openpose模型，一共25个关键点，一共是24段，在我们实际应用中可能不需要这么多
	Eigen::Vector3d Vec[24];
	//第1段
	Vec[0] = pt_world_arr[17] - pt_world_arr[15];
	len[0] = Vec[0].norm();
	//第2段
	Vec[1] = pt_world_arr[15] - pt_world_arr[0];
	len[1] = Vec[1].norm();
	//第3段
	Vec[2] = pt_world_arr[0] - pt_world_arr[16];
	len[2] = Vec[2].norm();
	//第4段
	Vec[3] = pt_world_arr[16] - pt_world_arr[18];
	len[3] = Vec[3].norm();
	//第5段
	Vec[4] = pt_world_arr[0] - pt_world_arr[1];
	len[4] = Vec[4].norm();
	//第6段
	Vec[5] = pt_world_arr[1] - pt_world_arr[2];
	len[5] = Vec[5].norm();
	//第7段
	Vec[6] = pt_world_arr[2] - pt_world_arr[3];
	len[6] = Vec[6].norm();
	//第8段
	Vec[7] = pt_world_arr[3] - pt_world_arr[4];
	len[7] = Vec[7].norm();
	//第9段
	Vec[8] = pt_world_arr[1] - pt_world_arr[5];
	len[8] = Vec[8].norm();
	//第10段
	Vec[9] = pt_world_arr[5] - pt_world_arr[6];
	len[9] = Vec[9].norm();
	//第11段
	Vec[10] = pt_world_arr[6] - pt_world_arr[7];
	len[10] = Vec[10].norm();
	//第12段
	Vec[11] = pt_world_arr[1] - pt_world_arr[8];
	len[11] = Vec[11].norm();
	//第13段
	Vec[12] = pt_world_arr[8] - pt_world_arr[9];
	len[12] = Vec[12].norm();
	//第14段
	Vec[13] = pt_world_arr[9] - pt_world_arr[10];
	len[13] = Vec[13].norm();
	//第15段
	Vec[14] = pt_world_arr[10] - pt_world_arr[11];
	len[14] = Vec[14].norm();
	//第16段
	Vec[15] = pt_world_arr[11] - pt_world_arr[22];
	len[15] = Vec[15].norm();
	//第17段
	Vec[16] = pt_world_arr[8] - pt_world_arr[12];
	len[16] = Vec[16].norm();
	//第18段
	Vec[17] = pt_world_arr[12] - pt_world_arr[13];
	len[17] = Vec[17].norm();
	//第19段
	Vec[18] = pt_world_arr[13] - pt_world_arr[14];
	len[18] = Vec[18].norm();
	//第20段
	Vec[19] = pt_world_arr[14] - pt_world_arr[20];
	len[19] = Vec[19].norm();

	Vec[20] = pt_world_arr[13];
	//打印调试信息
	/*for (size_t i = 0; i < LenthCount; i++)
	{
		std::cout << "距离" << i + 1 << "为：" << len[i];
	}*/
	//return len;
	float radian_angle[6] = { 0 };
	//13、14两向量夹角 髋关节1
	radian_angle[0] = atan2(Vec[11].cross(Vec[13]).norm(), Vec[11].transpose() * Vec[13]);
	angle[0] = radian_angle[0] * 180 / M_PI;
	//14、15两向量夹角 膝关节1
	radian_angle[1] = atan2(Vec[13].cross(Vec[14]).norm(), Vec[13].transpose() * Vec[14]);
	angle[1] = radian_angle[1] * 180 / M_PI;
	//15、16两向量夹角 踝关节1
	radian_angle[2] = atan2(Vec[14].cross(Vec[15]).norm(), Vec[14].transpose() * Vec[15]);
	angle[2] = radian_angle[2] * 180 / M_PI;
	//17、18两向量夹角 髋关节2
	radian_angle[3] = atan2(Vec[11].cross(Vec[17]).norm(), Vec[11].transpose() * Vec[17]);
	angle[3] = radian_angle[3] * 180 / M_PI;
	//18、19两向量夹角 膝关节2
	radian_angle[4] = atan2(Vec[17].cross(Vec[18]).norm(), Vec[17].transpose() * Vec[18]);
	angle[4] = radian_angle[4] * 180 / M_PI;
	//14、15两向量夹角 踝关节2
	radian_angle[5] = atan2(Vec[18].cross(Vec[19]).norm(), Vec[18].transpose() * Vec[19]);
	angle[5] = radian_angle[5] * 180 / M_PI;
}


//获取深度图像像素对应于长度单位（米）的转换比例
float get_depth_scale(rs2::device dev)
{
	// Go over the device's sensors
	for (rs2::sensor& sensor : dev.query_sensors())
	{
		// Check if the sensor if a depth sensor
		if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
		{
			return dpt.get_depth_scale();
		}
	}
	throw std::runtime_error("Device does not have a depth sensor");
}

//判断图像流是否发生改变
bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
	for (auto&& sp : prev)
	{
		//If previous profile is in current (maybe just added another)
		auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
		if (itr == std::end(current)) //If it previous stream wasn't found in current
		{
			return true;
		}
	}
	return false;
}

//长度&角度数据写入文件
void writeData(float* len, float* angle)
{
	std::string pathName = "F:\\Yan_2\\robotContest\\finaldatacollect\\finalDataC\\Len_Angle\\";
	//获取当前时间用于文件命名
	time_t curTime = time(0);
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%Y_%m_%d", localtime(&curTime));
	//strftime(tmp, sizeof(tmp), "%Y_%m_%d_%H_%M_%S", localtime(&curTime));

	pathName.append(tmp);
	pathName.append("_data.txt");
	//std::cout << pathName << std::endl;
	std::ofstream out;
	out.open(pathName, std::ios::app);	//这里的5表示追加写入，而不是擦除写入
	if (!out.is_open())
		std::cout << "out文件创建\\打开失败，无法进行写入\n";
	//std::ofstream out("写入.txt");
	//右腿长度len[13、14、15] 右腿角度angle[0、1、2]
	/*for (size_t i = 0; i < LenthCount; i++)
	{
		out << len[i] << " ";
	}
	for (size_t i = 0; i < AngelCount; i++)
	{
		out << angle[i] << " ";
	}*/
	for (size_t i = 13; i < 16; i++)
	{
		out << len[i] << " ";
	}
	for (size_t i = 0; i < 3; i++)
	{
		out << angle[i] << " ";
	}
	//测试一秒保存多少组数据
	char tmp2[64];
	strftime(tmp2, sizeof(tmp2), "%Y_%m_%d_%H_%M_%S", localtime(&curTime));
	out << tmp2;
	out << "\n";
	out.close();
}
#if 1
//这个configureWrapper和OpenPoseDemo中的有些许差异，但是具体有什么区别不太清楚
void configureWrapper(op::Wrapper& opWrapper)
{
	try
	{
		// Configuring OpenPose

		// logging_level
		op::checkBool(
			0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
			__LINE__, __FUNCTION__, __FILE__);
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::Profiler::setDefaultX(FLAGS_profile_speed);

		// Applying user defined configuration - GFlags to program variables
		// outputSize
		const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");//强制使用输入的分辨率？
		//const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "848x480");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
		// faceNetInputSize
		const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
		// handNetInputSize
		const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
		// poseMode
		const auto poseMode = op::flagsToPoseMode(FLAGS_body);
		// poseModel
		const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
		// JSON saving
		if (!FLAGS_write_keypoint.empty())
			op::opLog(
				"Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
				" instead.", op::Priority::Max);
		// keypointScaleMode
		const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
		// heatmaps to add
		const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
			FLAGS_heatmaps_add_PAFs);
		const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
		// >1 camera view?
		const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
		// Face and hand detectors
		const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
		const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
		// Enabling Google Logging
		const bool enableGoogleLogging = true;

		// Pose configuration (use WrapperStructPose{} for default and recommended configuration)
		const op::WrapperStructPose wrapperStructPose{
			poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
			FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
			op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
			(float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
			heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
			FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
			op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging };
		opWrapper.configure(wrapperStructPose);
		// Face configuration (use op::WrapperStructFace{} to disable it)
		const op::WrapperStructFace wrapperStructFace{
			FLAGS_face, faceDetector, faceNetInputSize,
			op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
			(float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold };
		opWrapper.configure(wrapperStructFace);
		// Hand configuration (use op::WrapperStructHand{} to disable it)
		const op::WrapperStructHand wrapperStructHand{
			FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
			op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
			(float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold };
		opWrapper.configure(wrapperStructHand);
		// Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
		const op::WrapperStructExtra wrapperStructExtra{
			FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads };
		opWrapper.configure(wrapperStructExtra);
		// Output (comment or use default argument to disable any output)
		const op::WrapperStructOutput wrapperStructOutput{
			FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
			op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
			FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
			op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
			op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
			op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
			op::String(FLAGS_udp_port) };
		opWrapper.configure(wrapperStructOutput);
		// No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
		// Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
		if (FLAGS_disable_multi_thread)
			opWrapper.disableMultiThreading();
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}
#endif
//打印关键点信息函数
void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
	try
	{
		// Example: How to use the pose keypoints
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			op::opLog("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
			op::opLog("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
			op::opLog("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
			op::opLog("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
		}
		else
			op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}


}


//人体骨架显示函数
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, const char* color_win)
{
	try
	{
		// User's displaying/saving/other processing here
			// datum.cvOutputData: rendered frame with pose or heatmaps
			// datum.poseKeypoints: Array<float> with the estimated pose
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			// Display image
			const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
			if (!cvMat.empty())
			{
				//cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
				cv::imshow(color_win, cvMat);
				//这里cvMat的分辨率也是848x480 与colorframe和deptframe一致
				//std::cout << cvMat.size() <<"gnb!!!!!!!!!!!!!"<< std::endl;
				cv::waitKey(1);
			}
			else
				op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
		}
		else
			op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

//该函数是用来绘制动态曲线的
void Dialog1::Draw1(CDC* pDC, CRect& rectPicture, float* m_value)
{
	float fDeltaX;     // x轴相邻两个绘图点的坐标距离   
	float fDeltaY;     // y轴每个逻辑单位对应的坐标值   
	int nX;      // 在连线时用于存储绘图点的横坐标   
	int nY;      // 在连线时用于存储绘图点的纵坐标   
	CPen newPen;       // 用于创建新画笔   
	CPen* pOldPen;     // 用于存放旧画笔   
	CBrush newBrush;   // 用于创建新画刷   
	CBrush* pOldBrush; // 用于存放旧画刷   

	// 计算fDeltaX和fDeltaY   
	fDeltaX = (float)rectPicture.Width() / (POINT_COUNT - 1);
	//范围这里需要更改
	fDeltaY = (float)rectPicture.Height() / 180;//分成70份，理由是人体正常步态关节角变化在70°以内 变化最大的关节时

	newBrush.CreateSolidBrush(RGB(0, 0, 0)); // 创建黑色新画刷
	pOldBrush = pDC->SelectObject(&newBrush);// 选择新画刷，并将旧画刷的指针保存到pOldBrush   
	pDC->Rectangle(rectPicture);    // 以黑色画刷为绘图控件填充黑色，形成黑色背景   
	pDC->SelectObject(pOldBrush);    // 恢复旧画刷
	newBrush.DeleteObject();    // 删除新画刷


	newPen.CreatePen(PS_SOLID, 1, RGB(255, 0, 0));    // 创建实心画笔，粗度为1，颜色为红色
	pOldPen = pDC->SelectObject(&newPen);    // 选择新画笔，并将旧画笔的指针保存到pOldPen
	//下面这两句时绘制边界线的
	pDC->MoveTo(0, rectPicture.Width() / 70 * 5);//这里这个计算数值是把y轴分成了70份（70°的幅值），65°是极限角度，超过65报警
	pDC->LineTo(rectPicture.Width(), rectPicture.Width() / 70 * 5);

	newPen.CreatePen(PS_SOLID, 1, RGB(0, 255, 0));    // 创建实心画笔，粗度为1，颜色为绿色
	pOldPen = pDC->SelectObject(&newPen);    // 选择新画笔，并将旧画笔的指针保存到pOldPen
	pDC->MoveTo(rectPicture.left, rectPicture.bottom);    // 将当前点移动到绘图控件窗口的左下角，以此为波形的起始点   
	for (int i = 0; i < POINT_COUNT; i++)    // 计算m_nzValues数组中每个点对应的坐标位置，并依次连接，最终形成曲线   
	{
		nX = rectPicture.left + (int)(i * fDeltaX);
		nY = rectPicture.bottom - (int)(m_value[i] * fDeltaY);
		//nY = 200;
		//CString strrr;
		//strrr.Format(_T("%f"), nY);
		//n_edit3.SetWindowTextW(strrr);

		pDC->LineTo(nX, nY);
	}

	////添加轴
	////即角度、时间的坐标
	//CFont font;//设置显示字体
	//font.CreateFont(13,                                    //   字体的高度   
	//			0,                                          //   字体的宽度  
	//			0,                                          //  nEscapement 
	//			0,                                          //  nOrientation   
	//			FW_NORMAL,                                  //   nWeight   
	//			FALSE,                                      //   bItalic   
	//			FALSE,                                      //   bUnderline   
	//			0,                                                   //   cStrikeOut   
	//			ANSI_CHARSET,                             //   nCharSet   
	//			OUT_DEFAULT_PRECIS,                 //   nOutPrecision   
	//			CLIP_DEFAULT_PRECIS,               //   nClipPrecision   
	//			DEFAULT_QUALITY,                       //   nQuality   
	//			DEFAULT_PITCH | FF_SWISS,     //   nPitchAndFamily     
	//			_T("宋体"));
	//pDC->SelectObject(&font);
	//pDC->SetBkMode(TRANSPARENT);
	//pDC->SetTextColor(RGB(255, 0, 0));
	////经过测试，130大概就是图像
	//pDC->TextOutW(-30, 0, _T("180°"));
	//pDC->TextOutW(-30, 65, _T("90°"));
	//pDC->TextOutW(-30, 130, _T("0°"));

	pDC->SelectObject(pOldPen);    // 恢复旧画笔   
	newPen.DeleteObject();    // 删除新画笔   
}

void MatToCImage(cv::Mat& mat, CImage& cimage)
{
	if (0 == mat.total())
	{
		return;
	}
	int nChannels = mat.channels();
	if ((1 != nChannels) && (3 != nChannels))
	{
		return;
	}
	int nWidth = mat.cols;//对应宽度 x
	int nHeight = mat.rows;//对应高度 y

	//重建cimage
	cimage.Destroy();
	cimage.Create(nWidth, nHeight, 8 * nChannels);

	//拷贝数据
	uchar* pucRow;									//指向数据区的行指针
	uchar* pucImage = (uchar*)cimage.GetBits();		//指向数据区的指针
	int nStep = cimage.GetPitch();					//每行的字节数,注意这个返回值有正有负

	if (1 == nChannels)								//对于单通道的图像需要初始化调色板
	{
		RGBQUAD* rgbquadColorTable;
		int nMaxColors = 256;
		rgbquadColorTable = new RGBQUAD[nMaxColors];
		cimage.GetColorTable(0, nMaxColors, rgbquadColorTable);
		for (int nColor = 0; nColor < nMaxColors; nColor++)
		{
			rgbquadColorTable[nColor].rgbBlue = (uchar)nColor;
			rgbquadColorTable[nColor].rgbGreen = (uchar)nColor;
			rgbquadColorTable[nColor].rgbRed = (uchar)nColor;
		}
		cimage.SetColorTable(0, nMaxColors, rgbquadColorTable);
		delete[]rgbquadColorTable;
	}


	for (int nRow = 0; nRow < nHeight; nRow++)
	{
		pucRow = (mat.ptr<uchar>(nRow));
		for (int nCol = 0; nCol < nWidth; nCol++)
		{
			if (1 == nChannels)
			{
				*(pucImage + nRow * nStep + nCol) = pucRow[nCol];
			}
			else if (3 == nChannels)
			{
				for (int nCha = 0; nCha < 3; nCha++)
				{
					*(pucImage + nRow * nStep + nCol * 3 + nCha) = pucRow[nCol * 3 + nCha];
				}
			}
		}
	}
}

void CImageToMat(CImage& cimage, cv::Mat& mat)
{
	if (true == cimage.IsNull())
	{
		return;
	}


	int nChannels = cimage.GetBPP() / 8;
	if ((1 != nChannels) && (3 != nChannels))
	{
		return;
	}
	int nWidth = cimage.GetWidth();
	int nHeight = cimage.GetHeight();


	//重建mat
	if (1 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC1);
	}
	else if (3 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC3);
	}


	//拷贝数据
	uchar* pucRow;									//指向数据区的行指针
	uchar* pucImage = (uchar*)cimage.GetBits();		//指向数据区的指针
	int nStep = cimage.GetPitch();					//每行的字节数,注意这个返回值有正有负


	for (int nRow = 0; nRow < nHeight; nRow++)
	{
		pucRow = (mat.ptr<uchar>(nRow));
		for (int nCol = 0; nCol < nWidth; nCol++)
		{
			if (1 == nChannels)
			{
				pucRow[nCol] = *(pucImage + nRow * nStep + nCol);
			}
			else if (3 == nChannels)
			{
				for (int nCha = 0; nCha < 3; nCha++)
				{
					pucRow[nCol * 3 + nCha] = *(pucImage + nRow * nStep + nCol * 3 + nCha);
				}
			}
		}
	}
}

// Dialog1 对话框

IMPLEMENT_DYNAMIC(Dialog1, CDialogEx)

Dialog1::Dialog1(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIALOG1, pParent)
{

}

Dialog1::~Dialog1()
{
}

void Dialog1::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT1, m_edit1);
	DDX_Control(pDX, IDC_EDIT3, m_edit2);
	DDX_Control(pDX, IDC_EDIT5, m_edit3);
	DDX_Control(pDX, IDC_EDIT2, m_len1);
	DDX_Control(pDX, IDC_EDIT4, m_len2);
	DDX_Control(pDX, IDC_EDIT6, m_len3);
	DDX_Control(pDX, IDC_STATIC1, m_draw1);
	DDX_Control(pDX, IDC_STATIC2, m_draw2);
	DDX_Control(pDX, IDC_STATIC3, m_draw3);
	srand((unsigned)time(NULL));   //以时间为种子来构造随机数生成器 
	SetTimer(1, 50, NULL);//启动定时器，ID为1，定时时间为50ms
}


BEGIN_MESSAGE_MAP(Dialog1, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON1, &Dialog1::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &Dialog1::OnBnClickedButton2)
	ON_WM_TIMER()
END_MESSAGE_MAP()


// Dialog1 消息处理程序


void Dialog1::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	// 显示图像
		//初始化部分：realsense相机的初始化 以及openpose的初始化
		//*****************************realsense初始化*****************************
		// Create a Pipeline
		rs2::pipeline pipe;
	// Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;
	// Configure and start the pipeline
	rs2::config pipe_config;
	pipe_config.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 60);//60是每秒帧数
	pipe_config.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_RGB8, 60);
	rs2::pipeline_profile profile;
	//打开相机
	profile = pipe.start(pipe_config);
	cv::waitKey(2000);
	//创建视频窗口
	//const char* depth_win = "depth_Image";
	//const char* color_win = "color_Image";
	//cv::namedWindow(depth_win, cv::WINDOW_AUTOSIZE);
	//cv::namedWindow(color_win, cv::WINDOW_AUTOSIZE);
	//使用数据管道的profile获取深度图像像素对应于长度单位（米）的转换比例
	float depth_scale = get_depth_scale(profile.get_device());
	//选择彩色图像数据流来作为对齐对象
	rs2_stream align_to = RS2_STREAM_COLOR;//find_stream_to_align(profile.get_stream());
	//创建一个rs2::align的对象
	//rs2::align 允许我们去实现深度图像对齐其他图像
	// "align_to"是我们打算用深度图像对齐的图像流
	rs2::align align(align_to);
	//定义一个变量去转换深度到距离
	float depth_clipping_distance = 1.f;
	//*****************************openpose初始化*****************************
	op::opLog("Starting OpenPose demo...", op::Priority::High);//简单理解就是打印信息函数，打印出“”中的字符串
	const auto opTimer = op::getTimerInit();//获取当前时间
	FLAGS_number_people_max = 1;//最大识别人数
	FLAGS_net_resolution = "320x176";
	// Configuring OpenPose
	op::opLog("Configuring OpenPose...", op::Priority::High);

	//感觉wrapper就类似于realsense中的pipeline 
	op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };	// typedef WrapperT<BASE_DATUM> Wrapper;
	//参数配置														
	configureWrapper(opWrapper);

	// Starting OpenPose
	op::opLog("Starting thread(s)...", op::Priority::High);
	opWrapper.start();

	CImage cimage;//用来转换MAT格式，方便在MFC窗口中显示
	//下面是视频流循环部分，realsense和openpose的初始话部分应该在这前面完成
	//while (cvGetWindowHandle(depth_win) )

	//滤波器初始化
	KalmanInfo kal[3];

	Init_KalmanInfo(&kal[0], 0.01, 0.1);
	Init_KalmanInfo(&kal[1], 0.01, 0.1);
	Init_KalmanInfo(&kal[2], 0.01, 0.1);

	while (1)
	{
		rs2::frameset data = pipe.wait_for_frames();			// 等待下一帧
		//因为rs2::align 正在对齐深度图像到其他图像流，要确保对齐的图像流不发生改变
		if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
		{
			//如果profile发生改变，则更新align对象，重新获取深度图像像素到长度单位的转换比例
			profile = pipe.get_active_profile();
			align = rs2::align(align_to);
			depth_scale = get_depth_scale(profile.get_device());
		}

		//获取对齐后的帧
		auto processed = align.process(data);
		rs2::depth_frame depth_frame = processed.get_depth_frame();
		rs2::frame depth = depth_frame.apply_filter(color_map);	//获取深度图，加颜色滤镜
		rs2::frame color = processed.get_color_frame();				//获取彩色图

		// 获取depth 和 color的像素尺寸
		const int depth_w = depth.as<rs2::video_frame>().get_width();
		const int depth_h = depth.as<rs2::video_frame>().get_height();
		//这里可以注销
		const int color_w = color.as<rs2::video_frame>().get_width();
		const int color_h = color.as<rs2::video_frame>().get_height();
		if (!depth || !color)
		{
			continue;
		}

		// cv格式的Mat接受图像数据
		//一般的图像文件格式使用的是 Unsigned 8bits吧，CvMat矩阵对应的参数类型就是CV_8UC1，CV_8UC2，CV_8UC3。（最后的1、2、3表示通道数，譬如RGB3通道就用CV_8UC3）
		//size cv矩阵类型3通道 数据源 步长
		cv::Mat depth_image(cv::Size(depth_w, depth_h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
		//cv::Mat color_image(cv::Size(color_w, color_h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat color_image(cv::Size(depth_w, depth_h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
		cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);//将BGR图像转化为RGB图像

		const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(color_image);//将cvMat转化为opMatrix类型 
		auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
		if (datumProcessed != nullptr)
		{
			//后面的计算任务也可以到printKeypoints函数里完成，或再单独写个函数在里面调用
			//printKeypoints(datumProcessed);
			if (!FLAGS_no_display)
			{
				//display(datumProcessed, color_win);
				if (!datumProcessed->empty())
				{
					// Display image
					const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
					if (!cvMat.empty())
					{
						CWnd* pWnd = GetDlgItem(IDC_STATIC);//获得pictrue控件窗口的句柄 
						CDC* pDC = pWnd->GetDC();//获得pictrue控件的DC  
						HDC hDC = pDC->GetSafeHdc();
						CRect rect;
						GetClientRect(&rect);
						GetDlgItem(IDC_STATIC)->GetClientRect(&rect);
						cv::Mat dst;
						int x = rect.Width();
						int y = rect.Height();
						resize(cvMat, dst, cv::Size(x, y));
						MatToCImage(dst, cimage);
						//下面这个函数相当于是绘制picture的
						cimage.Draw(pDC->m_hDC, rect);
						//cv::imshow(color_win, cvMat);
						//这里cvMat的分辨率也是848x480 与colorframe和deptframe一致
						//std::cout << cvMat.size() <<"gnb!!!!!!!!!!!!!"<< std::endl;
						//cv::waitKey(1);
					}
					else
						op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
				}
			}
			//获取像素坐标
			//piexl是个二维数组 25*3 25个关键点、x y score 
			//后面我们可能用不到这么多，可以根据需要减少数据量，降低程序开销
			int** pixel = getPixel(datumProcessed);

			//在这里插入卡尔曼滤波程序，对像素坐标进行滤波处理
			//改变思路：因为人在图像中的位置是不确定的，对像素坐标进行滤波不太合理，应该对求出的肢体参数、关节角度进行滤波处理

			//调试打印像素坐标信息
			/*for (size_t i = 0; i < 25; i++)
			{
				std::cout <<"x = "<< pixel[i][0] <<"	y = "<< pixel[i][1] << "gnb!!!!!!!!!!!!!" << std::endl;
			}*/

			//drawBody(pixel, drawBodyMat);//根据像素坐标重新绘制棍棒图
			//计算深度信息
			float* depthData = getDepth(pixel, depth_frame);
			/*
			//深度frame的分辨率
			int x = depth_frame.get_width();
			int y = depth_frame.get_height();
			std::cout << x << "	" << y << std::endl;
			*/

			//实现三维重建
			Eigen::Vector3d* pt_world3d;
			pt_world3d = (Eigen::Vector3d*)malloc(25 * sizeof(Eigen::Vector3d));
			loc3d(pixel, depthData, pt_world3d);
			/*std::cout << "三维坐标: " << pt_world3d[0][0] << " , " << pt_world3d[0][1] << " , " << pt_world3d[0][2] << std::endl;
			std::cout << "三维坐标: " << pt_world3d[15][0] << " , " << pt_world3d[15][1] << " , " << pt_world3d[15][2] << std::endl;
			std::cout << "三维坐标: " << pt_world3d[16][0] << " , " << pt_world3d[16][1] << " , " << pt_world3d[16][2] << std::endl;
			*/
			//计算肢体长度和关节角
			//float* len, * angle;
			len = (float*)malloc(24 * sizeof(float));
			angle = (float*)malloc(6 * sizeof(float));
			calculateDisAngle(pt_world3d, len, angle);
			//调试打印长度
			/*for (size_t i = 0; i < 24; i++)
			{
				std::cout << "第" << i << "段长为：" << len[i] << std::endl;
			}*/

			//滤波
			//double* KalmanFs(double* lastMeasurement, KalmanInfo * *info)

			KalmanFs(angle, kal);

			//保存数据
			writeData(len, angle);
			//writeCoordinate2txt(pt_world3d);

			//将数据显示到Edit中
			//n_edit1.SetWindowTextW(TEXT("aaa"));
			CString Edit1, Edit2, Edit3, len1, len2, len3;
			Edit1.Format(_T("%f"), angle[0]);
			Edit2.Format(_T("%f"), angle[1]);
			Edit3.Format(_T("%f"), angle[2]);
			len1.Format(_T("%f"), len[13] * 1000);//乘1000转化为mm
			len2.Format(_T("%f"), len[14] * 1000);
			len3.Format(_T("%f"), len[15] * 1000);
			m_edit1.SetWindowTextW(Edit1);
			m_edit2.SetWindowTextW(Edit2);
			m_edit3.SetWindowTextW(Edit3);
			m_len1.SetWindowTextW(len1);
			m_len2.SetWindowTextW(len2);
			m_len3.SetWindowTextW(len3);


		}
		else
			continue;
		//cv::imshow(depth_win, depth_image);
		//cv::imshow(color_win, color_image);
		//cv::destroyWindow(color_win);
		cv::waitKey(1);

	}
}


void Dialog1::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	cv::destroyAllWindows();
	CDialogEx::OnCancel();
	exit(0);
}


void Dialog1::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	CRect rectPicture1, rectPicture2, rectPicture3;
	// 将数组中的所有元素前移一个单位，第一个元素丢弃   
	for (int i = 0; i < POINT_COUNT - 1; i++)
	{
		m_value1[i] = m_value1[i + 1];
		m_value2[i] = m_value2[i + 1];
		m_value3[i] = m_value3[i + 1];
	}
	// 为最后一个元素赋一个80以内的随机数值（整型）   
	//m_value1[POINT_COUNT - 1] = rand() % 80;
	if (angle != NULL)
	{
		m_value1[POINT_COUNT - 1] = angle[0];//有bug
		m_value2[POINT_COUNT - 1] = angle[1];//有bug
		m_value3[POINT_COUNT - 1] = angle[2];//有bug
	}

	// 获取绘图控件的客户区坐标
	// （客户区坐标以窗口的左上角为原点，这区别于以屏幕左上角为原点的屏幕坐标）   
	m_draw1.GetClientRect(&rectPicture1);
	m_draw2.GetClientRect(&rectPicture2);
	m_draw3.GetClientRect(&rectPicture3);
	// 绘制波形图
	Draw1(m_draw1.GetDC(), rectPicture1, m_value1);
	Draw1(m_draw2.GetDC(), rectPicture2, m_value2);
	Draw1(m_draw3.GetDC(), rectPicture3, m_value3);
	CDialogEx::OnTimer(nIDEvent);
}
