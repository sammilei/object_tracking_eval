#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tldDataset.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <dirent.h>
#include <sys/types.h>

#include <fstream>

#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

bool debug = true;
bool debug2 = true;

unsigned int microseconds = 350000;
int img_w = 640;
int img_h = 480;

vector<string> bbox_files;

/**
 * for evaluation
 */
vector<float> ave_iou;
vector<float> frame_count;
vector<int> detected_frame_count;
vector<float> detected_rate;
vector<float> reinitial_bbox_area;
float iou_threshold;

vector<std::string> stringSplit(std::string line, std::string delimiter) {
  vector<std::string> tokens;
  size_t pos = 0;
  std::string token;
  while ((pos = line.find(delimiter)) != std::string::npos) {
    token = line.substr(0, pos);
    line.erase(0, pos + delimiter.length());
    tokens.push_back(token);
  }
  tokens.push_back(line);

  return tokens;
}

int dirExists(const char* path) {
  struct stat info;

  if (stat(path, &info) != 0)
    return 0;
  else if (info.st_mode & S_IFDIR)
    return 1;
  else
    return 0;
}

string addLeadingZero(int i, int len) {
  string s = to_string(i);
  int len_left = len - s.size();
  for (int k = 0; k < len_left; k++) {
    s = "0" + s;
  }
  return s;
}

vector<string> getFiles(std::string path) {
  DIR* dp;
  struct dirent* ep;
  vector<string> files;
  dp = opendir(path.c_str());
  if (dp != NULL) {
    while (ep = readdir(dp)) {
      char* output = NULL;
      output = strstr(ep->d_name, ".jpg");
      if (output) {
        files.push_back(ep->d_name);
      }
    }
    // puts (ep->d_name);
    (void)closedir(dp);
  } else
    perror("Couldn't open the directory");
  return files;
}

float calcIntersection(Rect2d boxA, Rect2d boxB) {
  // determine the (x, y)-coordinates of the intersection rectangle
  float xA = max(boxA.x, boxB.x);
  float yA = max(boxA.y, boxB.y);
  float xB = min(boxA.x + boxA.width, boxB.x + boxB.width);
  float yB = min(boxA.y + boxA.height, boxB.y + boxB.height);

  // compute the area of intersection rectangle
  float overLap = max(float(0), xB - xA + 1) * max(float(0), yB - yA + 1);

  return overLap;
}

float calcIoU(Rect2d boxA, Rect2d boxB) {
  float interArea = calcIntersection(boxA, boxB);
  if (debug2)
    cout << "interArea: " << interArea
         << " union:" << float(boxA.area() + boxB.area() - interArea) * 1.000
         << " ";

  float IoU = interArea / float(boxA.area() + boxB.area() - interArea);
  if (IoU > 1) {
    return 1;
  } else if (IoU < 0) {
    return 0;
  }

  return IoU;
}

Rect2d yoloToDbox(float x, float y, float w, float h) {
  Rect2d db;
  db.x = (x - 0.5 * w) * img_w;
  db.y = (y - 0.5 * h) * img_h;
  db.width = w * img_w;
  db.height = h * img_h;
  return db;
}

Rect2d getGTbb(string path_, std::string class_) {
  ifstream GT_file;
  Rect2d db;
  GT_file.open(path_);

  string line;
  if (GT_file.is_open()) {
    while (getline(GT_file, line)) {
      vector<string> in_a_line = stringSplit(line, " ");
      if(debug)
      cout << in_a_line << endl;
      if (in_a_line[0] == class_) {
        db = yoloToDbox(stof(in_a_line[1]),
                        stof(in_a_line[2]),
                        stof(in_a_line[3]),
                        stof(in_a_line[4]));
      }
    }
    GT_file.close();
  } else {
    std::cout << "no such a GT file: " << path_ << endl;
    exit(1);
  }
  return db;
}

vector<Rect2d> getReinitialBbox(string path_) {
  ifstream GT_file;
  vector<Rect2d> initial_bbox;
  GT_file.open(path_);

  string line;
  if (GT_file.is_open()) {
    while (getline(GT_file, line)) {
      vector<string> in_a_line = stringSplit(line, " ");
      bbox_files.push_back(in_a_line[0]);
      Rect2d db = yoloToDbox(stof(in_a_line[1]),
                             stof(in_a_line[2]),
                             stof(in_a_line[3]),
                             stof(in_a_line[4]));
      // Rect2d db(100, 200, 20, 30);
      initial_bbox.push_back(db);
    }
    GT_file.close();
  } else {
    std::cout << "no such a GT file: " << path_ << endl;
    exit(1);
  }
  return initial_bbox;
}

bool confirmTracker(string trackerType) {
  // List of tracker types in OpenCV 3.4.1
  string trackerTypes[8] = {
      "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

  for (int i = 0; i < 8; i++) {
    if (trackerTypes[i] == trackerType)
      return true;
  }
  return false;
}

Ptr<Tracker> createTracker(string trackerType) {
  Ptr<Tracker> tracker;

  if (trackerType == "BOOSTING")
    tracker = TrackerBoosting::create();
  if (trackerType == "MIL")
    tracker = TrackerMIL::create();
  if (trackerType == "KCF")
    tracker = TrackerKCF::create();
  if (trackerType == "TLD")
    tracker = TrackerTLD::create();
  if (trackerType == "MEDIANFLOW")
    tracker = TrackerMedianFlow::create();
  if (trackerType == "GOTURN")
    tracker = TrackerGOTURN::create();
  if (trackerType == "MOSSE")
    tracker = TrackerMOSSE::create();
  if (trackerType == "CSRT")
    tracker = TrackerCSRT::create();

  if (debug)
    std::cout << "create tracker\n";
  return tracker;
}

// Convert to string
#define SSTR(x)                                                                \
  static_cast<std::ostringstream&>((std::ostringstream() << std::dec << x))    \
      .str()

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << "argc: " << argc
              << " usage: ./Tracker <folder to images> <Tracker> <labels file> "
                 "<path_to_save_output_frames>"
                 "<class>"
                 "<iou threshold>\n";
    return -1;
  } else {
    cout << argv[1] << endl
         << argv[2] << endl
         << argv[3] << endl
         << argv[4] << endl
         << argv[5] << endl
         << argv[6] << endl;
  }

  // get the class
  string class_ = argv[5];

  // Read folder
  // string folder = "/home/parallels/Desktop/Parallels Shared
  // Folders/Home/Documents/JPL-Tech/artifact_reporting/Dataset/198_all/198_4cls_train_241/";
  string folder_to_images = argv[1];
  vector<string> files = getFiles(folder_to_images);
  if (files.size() == 0) {
    return -1;
  }

  string output_path = argv[4];
  if (!dirExists(output_path.c_str())) {
    cout << "ERROR: output path doesn't exist.\n";
    return -1;
  }

  // create a tracker
  if (!confirmTracker(argv[2])) {
    printf("Tracker is not found in OpenCV\n");
    exit;
  }
  string trackerType = argv[2];
  Ptr<Tracker> tracker;

  string bbox_path = argv[3];
  vector<Rect2d> bboxes = getReinitialBbox(bbox_path); // TODO: error exceptions

  try {
    iou_threshold = stof(argv[6]);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }

  // initialize a lot of vectors
  for (int i = 0; i < bboxes.size(); i++) {
    ave_iou.push_back(0);
    frame_count.push_back(0);
    detected_frame_count.push_back(0);
    detected_rate.push_back(0);
    reinitial_bbox_area.push_back(0);
  }

  Mat frame;
  Rect2d bbox;

  int count = 1; // display frame counts
  int reinitial_bbox_count = -1;

  float detected_total_fps = 0;
  int detected_total_count = 0;

  for (int i = 0; i < files.size(); i++) {
    if (debug)
      std::cout << "\n\ni: " << i << endl;
    bool init = false;
    bool ok = false;

    double timer = (double)getTickCount();
    float fps;

    frame = imread(folder_to_images + '/' + files[i], 1);

    float IoU;

  // get the GT bbox
  std:
    string file_name_base = stringSplit(files[i], ".")[0];
    Rect2d GT_bb =
        getGTbb(folder_to_images + '/' + file_name_base + ".txt", class_);

    for (int num_f = reinitial_bbox_count + 1; num_f < bbox_files.size();
         num_f++) {
      if (bbox_files[num_f] == files[i]) {
        if (debug)
          std::cout << bbox_files[num_f] << " vs " << files[i] << endl;
        bbox = bboxes[num_f];
        // re initial tracker
        tracker.release();
        tracker = createTracker(trackerType);
        ok = tracker->init(frame, bbox);
        init = true;
        ok = true;
        ++reinitial_bbox_count;
        // for evluation
        reinitial_bbox_area[reinitial_bbox_count] = bbox.area();
        break;
      }
    }

    if (!init) {
      if (debug2)
        cout << "in !init "
             << "\n";
      if (debug2)
        cout << "frame size: " << frame.size() << " bbox: (" << bbox.x << ", "
             << bbox.y << "), (" << bbox.width << ", " << bbox.height << ")"
             << endl;
      ok = tracker->update(frame, bbox);

      if (debug2)
        cout << "tracker bbox: (" << bbox.x << ", " << bbox.y << "), ("
             << bbox.width << ", " << bbox.height << ")" << endl;
      fps = getTickFrequency() / ((double)getTickCount() - timer);
      frame_count[reinitial_bbox_count] += 1;
      if (debug)
        std::cout << "x: " << bbox.x << " y:" << bbox.y << " w:" << bbox.width
                  << " h:" << bbox.height << endl;
      if (debug)
        std::cout << "update\n";
    } else {
      if (debug2)
        cout << "initial bbox: (" << bbox.x << ", " << bbox.y << "), ("
             << bbox.width << ", " << bbox.height << ")" << endl;

      if (debug)
        std::cout << "init\n";
      fps = 0;
    }

    if (debug)
      std::cout << "ok: " << ok << " ";

    if (ok) {
      if (init) {
        cv::putText(frame,
                    "Reinitial" + to_string(reinitial_bbox_count + 1),
                    Point(100, 20),
                    FONT_HERSHEY_SIMPLEX,
                    0.75,
                    Scalar(0, 255, 255),
                    2);
        rectangle(frame, bbox, Scalar(0, 255, 255), 2, 1);
      } else {
        // Tracking success : Draw the tracked object
        rectangle(frame, bbox, Scalar(0, 0, 255), 2, 1);
        // get the GT bbox
        IoU = (calcIoU(GT_bb, bbox));
        cout << " iou: " << IoU << " " << files[i] << endl;

        // for evaluation statistics
        if (IoU >= iou_threshold) {
          detected_total_fps += fps;
          detected_total_count++;
          ave_iou[reinitial_bbox_count] += IoU;
          detected_frame_count[reinitial_bbox_count] += 1;
        } else {
          cv::putText(frame,
                      "Tracked but IoU<" + to_string(iou_threshold),
                      Point(100, 20),
                      FONT_HERSHEY_SIMPLEX,
                      0.6,
                      Scalar(255, 0, 255),
                      2);
        }
      }
    } else {
      // Tracking failure detected.
      cv::putText(frame,
                  "Tracking failure detected",
                  Point(100, 20),
                  FONT_HERSHEY_SIMPLEX,
                  0.65,
                  Scalar(0, 0, 255),
                  2);
    }

    rectangle(frame, GT_bb, Scalar(0, 255, 255), 1, 1); // GT ground truth
    // Display FPS on frame
    cv::putText(frame,
                "FPS : " + SSTR(int(fps)),
                Point(320, 210),
                FONT_HERSHEY_SIMPLEX,
                0.60,
                Scalar(50, 170, 50),
                2);

    // Display frame
    cv::putText(frame,
                "#" + std::to_string(count) + "/" +
                    std::to_string(files.size()),
                Point(360, 230),
                FONT_HERSHEY_SIMPLEX,
                0.50,
                Scalar(255, 255, 25),
                1);

    cv::putText(frame,
                "IoU: " + std::to_string(IoU) + " " + files[i],
                Point(100, 230),
                FONT_HERSHEY_SIMPLEX,
                0.40,
                Scalar(255, 255, 255),
                1);
    count++;

    cv::imshow("Tracker " + trackerType, frame);
    string saved_name = output_path + "/" + trackerType + "iou" +
        to_string(iou_threshold) + " class" + class_ + "_no." +
        addLeadingZero(i, 3);
    if (init) {
      saved_name += "_reinitial.jpg";
    } else {
      saved_name += ".jpg";
    }
    cv::imwrite(saved_name, frame);
    // usleep(microseconds);

    // Exit if ESC pressed.
    int k = waitKey(1);
    if (k == 27) {
      break;
    }
  }

  // sort out the statistics
  int total_frames = 0;
  float total_reinitialize_area = 0;
  float total_ave_detected_rate;
  float total_ave_iou;
  for (int i = 0; i < frame_count.size(); i++) {
    ave_iou[i] /= frame_count[i];
    detected_rate[i] = detected_frame_count[i] / frame_count[i];
    total_frames += frame_count[i];
    total_reinitialize_area += reinitial_bbox_area[i];
    total_ave_detected_rate += detected_rate[i] * detected_frame_count[i];
    total_ave_iou += ave_iou[i] * detected_frame_count[i];
  }

  // report
  cout << "=================================================================\n";
  cout << "Report of Tracker: " << trackerType << endl;
  cout << "=================================================================\n";

  string classes[4] = {"fire extinguisher", "backpack", "drill", "survivor"};
  cout << "total frame: " << total_frames
       << "(except the reinitial frames), IoU threshold: " << iou_threshold
       << endl
       << "category: " << class_ << " (maybe: " << classes[stoi(class_)] << ")"
       << endl;

  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  for (int i = 0; i < frame_count.size(); i++) {
    cout << "size " << i << " --  pixel area: " << reinitial_bbox_area[i]
         << ", "
         << "detected rate: " << detected_rate[i]
         << ", average IoU: " << ave_iou[i] << ", frame:" << frame_count[i]
         << endl;
  }
  cout << "total "
       << " -- ave pixel area: "
       << total_reinitialize_area / reinitial_bbox_area.size()
       << ", detected rate: " << total_ave_detected_rate / total_frames
       << ", average IoU: " << total_ave_iou / total_frames
       << endl; // bug: average IoU goes huge sometimes but it will be fine
                // after a few runs
  cout << "average detection fps: " << detected_total_fps / detected_total_count
       << endl;
}