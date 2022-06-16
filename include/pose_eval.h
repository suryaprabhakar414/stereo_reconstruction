#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>
#include<io_util.h>

#include "matrix.h"


using namespace std;

// static parameter
// float lengths[] = {5,10,50,100,150,200,250,300,350,400};
float lengths[] = {100,200,300,400,500,600,700,800};
int32_t num_lengths = 8;

struct errors {
  int32_t first_frame;
  float   r_err;
  float   t_err;
  float   len;
  float   speed;
  errors (int32_t first_frame,float r_err,float t_err,float len,float speed) :
    first_frame(first_frame),r_err(r_err),t_err(t_err),len(len),speed(speed) {}
};

struct simple_error {
    float r_err;
    float t_err;
    simple_error(float r_err, float t_err) :
      r_err(r_err), t_err(t_err) {}
};

vector<Matrix> loadPoses(string file_name, unsigned long n_lines) {
  vector<Matrix> poses;
  unsigned long cnt = 0;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (n_lines == 0)
      n_lines = UINT64_MAX;
  if (!fp)
    return poses;
  while (!feof(fp) && cnt < n_lines) {
    Matrix P = Matrix::eye(4);
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                   &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                   &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3])==12) {
      poses.push_back(P);
      cnt += 1;
    }
  }
  fclose(fp);
  return poses;
}

vector<float> trajectoryDistances (vector<Matrix> &poses) {
  vector<float> dist;
  dist.push_back(0);
  for (int32_t i=1; i<poses.size(); i++) {
    Matrix P1 = poses[i-1];
    Matrix P2 = poses[i];
    float dx = P1.val[0][3]-P2.val[0][3];
    float dy = P1.val[1][3]-P2.val[1][3];
    float dz = P1.val[2][3]-P2.val[2][3];
    dist.push_back(dist[i-1]+sqrt(dx*dx+dy*dy+dz*dz));
  }
  return dist;
}

int32_t lastFrameFromSegmentLength(vector<float> &dist,int32_t first_frame,float len) {
  for (int32_t i=first_frame; i<dist.size(); i++)
    if (dist[i]>dist[first_frame]+len)
      return i;
  return -1;
}

inline float rotationError(Matrix &pose_error) {
  float a = pose_error.val[0][0];
  float b = pose_error.val[1][1];
  float c = pose_error.val[2][2];
  float d = 0.5*(a+b+c-1.0);
  return acos(max(min(d,1.0f),-1.0f));
}

inline float translationError(Matrix &pose_error) {
  float dx = pose_error.val[0][3];
  float dy = pose_error.val[1][3];
  float dz = pose_error.val[2][3];
  return sqrt(dx*dx+dy*dy+dz*dz);
}

void saveSimpleErrors(vector<simple_error> &err, string file_name) {
    // open file
    FILE *fp;
    fp = fopen(file_name.c_str(),"w");

    // write to file
    for (auto & it : err)
        fprintf(fp, "%f %f\n", it.r_err, it.t_err);

    // close file
    fclose(fp);
}

void saveSequenceErrors (vector<errors> &err,string file_name) {

  // open file
  FILE *fp;
  fp = fopen(file_name.c_str(),"w");

  // write to file
  for (auto & it : err)
    fprintf(fp,"%d %f %f %f %f\n",it.first_frame,it.r_err,it.t_err,it.len,it.speed);

  // close file
  fclose(fp);
}


void saveStats (vector<errors> err,string dir) {

  float t_err = 0;
  float r_err = 0;

  // for all errors do => compute sum of t_err, r_err
  for (vector<errors>::iterator it=err.begin(); it!=err.end(); it++) {
    t_err += it->t_err;
    r_err += it->r_err;
  }

  // open file
  FILE *fp = fopen((dir + "/stats.txt").c_str(),"w");

  // save errors
  float num = err.size();
  fprintf(fp,"%f %f\n",t_err/num,r_err/num);

  // close file
  fclose(fp);
}

vector<simple_error> calcSimpleError (vector<Matrix> &poses_gt, vector<Matrix> &poses_result) {
    vector<simple_error> err;
    err.emplace_back(0.0f, 0.0f);
    Matrix start_pose = poses_gt[0];
    for (int32_t i = 1; i < poses_result.size(); i += 1) {
        Matrix pose_gt = poses_gt[i];
        Matrix pose_res = poses_result[i];
        Matrix pose_error = Matrix::inv(pose_res * start_pose)*pose_gt;
        float r_err = rotationError(pose_error);
        float t_err = translationError(pose_error);
        err.emplace_back(r_err, t_err);
    }
    return err;
}

vector<errors> calcSequenceErrors (vector<Matrix> &poses_gt,vector<Matrix> &poses_result) {

  // error vector
  vector<errors> err;

  // parameters
  int32_t step_size = 10; // every second

  // pre-compute distances (from ground truth as reference)
  vector<float> dist = trajectoryDistances(poses_gt);

  // for all start positions do
  for (int32_t first_frame=0; first_frame<poses_gt.size(); first_frame+=step_size) {

    // for all segment lengths do
    for (int32_t i=0; i<num_lengths; i++) {

      // current length
      float len = lengths[i];

      // compute last frame
      int32_t last_frame = lastFrameFromSegmentLength(dist,first_frame,len);

      // continue, if sequence not long enough
      if (last_frame==-1)
        continue;

      // compute rotational and translational errors
      Matrix pose_delta_gt     = Matrix::inv(poses_gt[first_frame])*poses_gt[last_frame];
      Matrix pose_delta_result = Matrix::inv(poses_result[first_frame])*poses_result[last_frame];
      Matrix pose_error        = Matrix::inv(pose_delta_result)*pose_delta_gt;
      float r_err = rotationError(pose_error);
      float t_err = translationError(pose_error);

      // compute speed
      float num_frames = (float)(last_frame-first_frame+1);
      float speed = len/(0.1*num_frames);

      // write to file
      err.push_back(errors(first_frame,r_err/len,t_err/len,len,speed));
    }
  }

  // return error vector
  return err;
}





void pose_eval (string algorithm) {

    // ground truth and result directories
    string gt_dir         = "../images/odometry";
    string result_dir     = "../result/"+algorithm;
    string error_dir      = result_dir + "/errors/";


    // create output directories
    //system(("mkdir"+ result_dir).c_str());
    system(("mkdir -p " + error_dir).c_str());


    // total errors
    vector<errors> total_err;

    // for all sequences do


    // file name
    char file_name[256] ="00.txt";

    // read ground truth and result poses
    vector<Matrix> poses_result = loadPoses(result_dir + "/pose.txt", 0);
    unsigned long actual_lines = poses_result.size();
    assert(actual_lines > 0);
    vector<Matrix> poses_gt     = loadPoses(gt_dir + "/" + file_name, actual_lines);


    // compute sequence errors
//    vector<errors> seq_err = calcSequenceErrors(poses_gt, poses_result);
    vector<simple_error> seq_err = calcSimpleError(poses_gt, poses_result);
//    saveSequenceErrors(seq_err,error_dir + "/" + file_name);
    saveSimpleErrors(seq_err, error_dir + "/" + file_name);
    //saveStats(seq_err,result_dir);


}
