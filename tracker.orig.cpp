#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tldDataset.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include <sys/types.h>
#include <dirent.h>

using namespace cv;
using namespace std;

unsigned int microseconds = 350000;

vector<string> getFiles(std::string path){
  DIR *dp;
  struct dirent *ep;
  vector <string> files; 
  dp = opendir (path.c_str());
  if (dp != NULL)
    {
      while (ep = readdir (dp)){
        char *output = NULL;
        output = strstr (ep->d_name,".jpg");
        if(output) {
          printf("String Found");
          files.push_back(ep->d_name);
        }
      }
        // puts (ep->d_name);
      (void) closedir (dp);
    }
  else
    perror ("Couldn't open the directory");
  return files;
}

// Convert to string
#define SSTR(x)                                                                \
  static_cast<std::ostringstream&>((std::ostringstream() << std::dec << x))    \
      .str()

int main(int argc, char** argv) {
  if ( argc != 3 )
    {
        printf("usage: ./Tracker <video_Path> <Tracker>\n");
        return -1;
    }
  // Read video
  VideoCapture video(argv[1]);

  // Read folder
  string folder = "/home/parallels/Desktop/Parallels Shared Folders/Home/Documents/JPL-Tech/artifact_reporting/Dataset/198_all/198_4cls_train_241";
  vector<string> files = getFiles(folder);
  if(files.size() == 0){
    return -1;
  }
  for (int i = 0; i < files.size(); i++){
    cout << files[i] << endl;
  }

  // Exit if video is not opened
  if (!video.isOpened()) {
    cout << "Could not read video file" << endl;
    return 1;
  }


  // List of tracker types in OpenCV 3.4.1
  string trackerTypes[8] = {
      "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
  // vector <string> trackerTypes(types, std::end(types));

  // Create a tracker
  //string trackerType = trackerTypes[2];
  bool found = false;
  
  for (int i = 0; i < 8; i++){
    if (trackerTypes[i] == argv[2]){
      found = true;
    }
  }

  if(found == false){
    printf("Tracker is not found in OpenCV\n");
    return -1;
  }
  
  string trackerType = argv[2];

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
  Mat frame;
  bool ok = video.read(frame);

  // Define initial bounding box
  Rect2d bbox(287, 23, 86, 320);

  // Uncomment the line below to select a different bounding box
  bbox = selectROI(frame, false);
  // Display bounding box.
  rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

  imshow("Tracker " + trackerType, frame);
  tracker->init(frame, bbox);

  int count = 0;
  while (video.read(frame)) {
  
    // Start timer
    double timer = (double)getTickCount();

    // Update the tracking result
    bool ok = tracker->update(files[i], bbox);

    // Calculate Frames per second (FPS)
    float fps = getTickFrequency() / ((double)getTickCount() - timer);

    if (ok) {
      // Tracking success : Draw the tracked object
      rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
    } else {
      // Tracking failure detected.
      putText(frame,
              "Tracking failure detected",
              Point(100, 80),
              FONT_HERSHEY_SIMPLEX,
              0.75,
              Scalar(0, 0, 255),
              2);
    }

    // Display FPS on frame
    putText(frame,
            "FPS : " + SSTR(int(fps)),
            Point(100, 20),
            FONT_HERSHEY_SIMPLEX,
            0.75,
            Scalar(50, 170, 50),
            2);

    // Display frame.
    putText(frame,
            "#" + std::to_string(count),
            Point(360,230),
             FONT_HERSHEY_SIMPLEX,
            0.60,
            Scalar(255, 255, 25),
            1);
    count ++;
    
    imshow("Tracking", frame);
    // usleep(microseconds);

    // Exit if ESC pressed.
    int k = waitKey(1);
    if (k == 27) {
      break;
    }
  }
}