/** usage: 
$./Tracker /home/parallels/Desktop/Parallels\ Shared\ Folders/Home/Documents/JPL-Tech/artifact_reporting/Dataset/198_all/198_4cls_train_241/by_class/drill_47/ CSRT bboxes/198_drill.txt 
**/

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

unsigned int microseconds = 350000;
int img_w = 424;
int img_h = 240;

vector<std::string> stringSplit(std::string line) {
  vector<std::string> tokens;
  std::string delimiter = " ";
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

float bbIntersection(Rect2d boxA, Rect2d boxB) {
  // determine the (x, y)-coordinates of the intersection rectangle
  float xA = max(boxA.x, boxB.x);
  float yA = max(boxA.y, boxB.y);
  float xB = min(boxA.x + boxA.width, boxB.x + boxB.width);
  float yB = min(boxA.y + boxA.height, boxB.y + boxB.height);

  // compute the area of intersection rectangle
  float interArea = max(float(0), xB - xA + 1) * max(float(0), yB - yA + 1);

  return interArea;
}

Rect2d yoloToDbox(float x, float y, float w, float h) {
  Rect2d db;
  db.x = x * img_w - 0.5 * w;
  db.y = y * img_h - 0.5 * h;
  db.width = w * img_w;
  db.height = h * img_h;
  return db;
}

vector<Rect2d> getReinitialBbox(string path_) {
  ifstream GT_file;
  vector<Rect2d> initialBbox;
  GT_file.open(path_);

  string line;
  if (GT_file.is_open()) {
    while (getline(GT_file, line)) {
      vector<string> in_a_line = stringSplit(line);
      Rect2d db = yoloToDbox(stof(in_a_line[0]),
                             stof(in_a_line[1]),
                             stof(in_a_line[2]),
                             stof(in_a_line[3]));
      // Rect2d db(100, 200, 20, 30);
      initialBbox.push_back(db);
    }
    GT_file.close();
  } else {
    cout << "no such a GT file: " << path_ << endl;
    exit(1);
  }

  return initialBbox;
}

// Convert to string
#define SSTR(x)                                                                \
  static_cast<std::ostringstream&>((std::ostringstream() << std::dec << x))    \
      .str()

int main(int argc, char** argv) {
  if (argc != 4) {
    cout << "argc: " << argc
         << " usage: ./Tracker <folder to images> <Tracker> <labels file>\n";
    return -1;
  }

  // Read folder
  // string folder = "/home/parallels/Desktop/Parallels Shared
  // Folders/Home/Documents/JPL-Tech/artifact_reporting/Dataset/198_all/198_4cls_train_241/";
  string folder_to_images = argv[1];
  vector<string> files = getFiles(folder_to_images);
  if (files.size() == 0) {
    return -1;
  }
  // for (int i = 0; i < files.size(); i++) {
  //   cout << files[i] << endl;
  // }

  // List of tracker types in OpenCV 3.4.1
  string trackerTypes[8] = {
      "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
  // vector <string> trackerTypes(types, std::end(types));

  // confirm a tracker
  bool found = false;

  for (int i = 0; i < 8; i++) {
    if (trackerTypes[i] == argv[2]) {
      found = true;
    }
  }
  if (found == false) {
    printf("Tracker is not found in OpenCV\n");
    return -1;
  }
  string trackerType = argv[2];

  // create a tracker
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

  // Read first frame
  Mat frame = imread(folder_to_images + files[0], 1);

  // Define initial bounding box
  string bbox_path = argv[3];

  int count = 1; // frame counts
  vector<Rect2d> bboxes = getReinitialBbox(bbox_path);
  // for (int i = 0; i < bboxes.size(); i++) {
  //   cout << bboxes[i].x << endl;
  // }
  bool ok = false;

  Rect2d bbox;
  for (int i = 0; i < files.size(); i++) {
    double timer = (double)getTickCount();

    frame = imread(folder_to_images + '/' + files[i]);
    cout << "x: " << bbox.x << " y:" << bbox.y << " w:" << bbox.width << " h:" << bbox.height << endl;

    if (i == 0) {
      bbox = bboxes[0];
      ok = tracker->init(frame, bbox);
    } else {
      ok = tracker->update(frame, bbox);
    }

    float fps = getTickFrequency() / ((double)getTickCount() - timer);

    // Display FPS on frame
    putText(frame,
            "FPS : " + SSTR(int(fps)),
            Point(300, 200),
            FONT_HERSHEY_SIMPLEX,
            0.60,
            Scalar(50, 170, 50),
            2);

    // Display frame
    putText(frame,
            "#" + std::to_string(count),
            Point(360, 230),
            FONT_HERSHEY_SIMPLEX,
            0.60,
            Scalar(255, 255, 25),
            1);
    if (ok) {
      // Tracking success : Draw the tracked object
      rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

    } else {
      // Tracking failure detected.
      putText(frame,
              "Tracking failure detected",
              Point(100, 20),
              FONT_HERSHEY_SIMPLEX,
              0.75,
              Scalar(0, 0, 255),
              2);
    }
    count++;

    imshow("Tracker " + trackerType, frame);
    usleep(microseconds);

    // Exit if ESC pressed.
    int k = waitKey(1);
    if (k == 27) {
      break;
    }
  }
}