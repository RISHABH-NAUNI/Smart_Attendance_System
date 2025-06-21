package com.attendance;

import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Scanner;

public class FaceCapture {

    static {
        // Load OpenCV 4.5.5 native library
        System.loadLibrary("opencv_java455");
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter your name (user ID): ");
        String userId = scanner.nextLine().trim();

        if (userId.isEmpty()) {
            System.err.println("User name cannot be empty.");
            scanner.close();
            return;
        }

        int sampleCount = 0;
        int totalSamples = 25;

        // Folder to save face images
        String saveDir = "dataset/" + userId;
        new File(saveDir).mkdirs();

        // Load Haar cascade from resources
        URL xmlUrl = FaceCapture.class.getClassLoader().getResource("haarcascade_frontalface_default.xml");
        if (xmlUrl == null) {
            System.err.println("❌ Haar cascade XML not found in resources!");
            scanner.close();
            return;
        }

        String cascadePath;
        try {
            cascadePath = Paths.get(xmlUrl.toURI()).toString();
        } catch (Exception e) {
            System.err.println("❌ Error converting Haar cascade URL to path: " + e.getMessage());
            scanner.close();
            return;
        }

        CascadeClassifier faceDetector = new CascadeClassifier(cascadePath);
        if (faceDetector.empty()) {
            System.err.println("❌ Failed to load Haar cascade from: " + cascadePath);
            scanner.close();
            return;
        }

        // Start camera
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.err.println("❌ Could not open camera.");
            scanner.close();
            return;
        }

        Mat frame = new Mat();
        System.out.println("📸 Capturing face images for user: " + userId);

        while (sampleCount < totalSamples) {
            if (camera.read(frame)) {
                Mat gray = new Mat();
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

                MatOfRect faces = new MatOfRect();
                faceDetector.detectMultiScale(
                        gray,
                        faces,
                        1.1,
                        5,
                        0,
                        new Size(30, 30),
                        new Size()
                );

                for (Rect face : faces.toArray()) {
                    Imgproc.rectangle(frame, face.tl(), face.br(), new Scalar(0, 255, 0), 2);

                    // Crop and save the face image
                    Mat faceROI = new Mat(gray, face);
                    String filename = saveDir + "/face" + sampleCount + ".png";
                    boolean saved = Imgcodecs.imwrite(filename, faceROI);
                    if (saved) {
                        System.out.println("✅ Saved " + filename);
                        sampleCount++;
                    } else {
                        System.err.println("❌ Failed to save " + filename);
                    }

                    if (sampleCount >= totalSamples) {
                        break;
                    }
                }
            }
        }

        camera.release();
        scanner.close();
        System.out.println("🎉 Done capturing " + totalSamples + " face images for user: " + userId);
    }
}
