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

using namespace cv;
using namespace std;

bool debug = false;

unsigned int microseconds = 350000;
int img_w = 424;
int img_h = 240;

vector<string> bbox_files;

/**
 * for evaluation
 */
vector<float> ave_iou;
vector<float> ave_iou_count;
vector<int> detected_count;
vector<float> detected_rate;
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
  vector<Rect2d> initialBbox;
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
      initialBbox.push_back(db);
    }
    GT_file.close();
  } else {
    std::cout << "no such a GT file: " << path_ << endl;
    exit(1);
  }
  return initialBbox;
}

bool confirmTracker(string trackerType) {
  // List of tracker types in OpenCV 3.4.1
  string trackerTypes[8] = {
      "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

  bool found = false;

  for (int i = 0; i < 8; i++) {
    if (trackerTypes[i] == trackerType) {
      return found;
    }
  }
  if (found == false) {
    printf("Tracker is not found in OpenCV\n");
    exit;
  }
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
  if (argc != 6) {
    std::cout << "argc: " << argc
              << " usage: ./Tracker <folder to images> <Tracker> <labels file> "
                 "<class>"
                 "<iou threshold>\n";
    return -1;
  }

  // get the class
  string class_ = argv[4];

  // Read folder
  // string folder = "/home/parallels/Desktop/Parallels Shared
  // Folders/Home/Documents/JPL-Tech/artifact_reporting/Dataset/198_all/198_4cls_train_241/";
  string folder_to_images = argv[1];
  vector<string> files = getFiles(folder_to_images);
  if (files.size() == 0) {
    return -1;
  }

  // create a tracker
  confirmTracker(argv[2]);
  string trackerType = argv[2];
  Ptr<Tracker> tracker;

  string bbox_path = argv[3];
  vector<Rect2d> bboxes = getReinitialBbox(bbox_path); // TODO: error exceptions

  try {
    iou_threshold = stof(argv[5]);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }

  // initialize a lot of vectors
  for (int i = 0; i < bboxes.size(); i++) {
    ave_iou.push_back(0);
    ave_iou_count.push_back(0);
    detected_count.push_back(0);
    detected_rate.push_back(0);
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
        break;
      }
    }

    if (!init) {
      ok = tracker->update(frame, bbox);
      ave_iou_count[reinitial_bbox_count] += 1;
      if (debug)

        std::cout << "x: " << bbox.x << " y:" << bbox.y << " w:" << bbox.width
                  << " h:" << bbox.height << endl;
      if (debug)
        std::cout << "update\n";
    } else {
      if (debug)
        std::cout << "init\n";
    }

    float fps = getTickFrequency() / ((double)getTickCount() - timer);

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
                "#" + std::to_string(count),
                Point(370, 230),
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
    if (debug)
      std::cout << "ok: " << ok << " ";

    if (ok) {
      rectangle(frame, GT_bb, Scalar(0, 255, 0), 1, 1); // ground truth
      if (init) {
        cv::putText(frame,
                    "re initial",
                    Point(100, 20),
                    FONT_HERSHEY_SIMPLEX,
                    0.75,
                    Scalar(0, 255, 255),
                    2);
        rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
      } else {
        // Tracking success : Draw the tracked object
        rectangle(frame, bbox, Scalar(0, 0, 255), 2, 1);
        // get the origin bbox
        IoU = (calcIoU(GT_bb, bbox));
        cout << " iou: " << IoU << " " << files[i] << " init: " << init << endl;

        // for evaluation statistics
        if (IoU >= iou_threshold) {
          detected_total_fps += fps;
          detected_total_count++;
          ave_iou[reinitial_bbox_count] += IoU;
          detected_count[reinitial_bbox_count] += 1;
        }
      }

    } else {
      // Tracking failure detected.
      cv::putText(frame,
                  "Tracking failure detected",
                  Point(100, 20),
                  FONT_HERSHEY_SIMPLEX,
                  0.75,
                  Scalar(0, 0, 255),
                  2);
    }
    count++;

    cv::imshow("Tracker " + trackerType, frame);
    // usleep(microseconds);

    // Exit if ESC pressed.
    int k = waitKey(1);
    if (k == 27) {
      break;
    }
  }

  // sort out the statistics
  cout << "\n\nreport\n";
  for (int i = 0; i < ave_iou_count.size(); i++) {
    ave_iou[i] /= ave_iou_count[i];
    detected_rate[i] = detected_count[i] / ave_iou_count[i];
    cout << " detected_rate: " << detected_rate[i]
         << " for # of frames: " << ave_iou_count[i] << endl;
    cout << "ave iou: " << ave_iou[i]
         << " for # of frames: " << ave_iou_count[i] << endl;
    cout << "detected_count: " << detected_count[i];
  }

  // report

  cout << "average detection fps: " << detected_total_fps / detected_total_count
       << endl;
}