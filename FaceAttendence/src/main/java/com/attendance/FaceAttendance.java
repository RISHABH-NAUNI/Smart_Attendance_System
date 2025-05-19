package com.attendance;

import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.highgui.HighGui;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class FaceAttendance {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private CascadeClassifier faceDetector;
    private MultiLayerNetwork model;
    private Map<Integer, String> indexToLabel;

    public FaceAttendance(String haarCascadePath, MultiLayerNetwork model, Map<Integer, String> indexToLabel) {
        faceDetector = new CascadeClassifier(haarCascadePath);
        if (faceDetector.empty()) {
            throw new IllegalStateException("Failed to load Haar cascade from " + haarCascadePath);
        }
        this.model = model;
        this.indexToLabel = indexToLabel;
    }

    public void startAttendance() {
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.err.println("❌ Could not open camera.");
            return;
        }

        Mat frame = new Mat();
        System.out.println("🎥 Starting attendance. Press 'q' to quit.");

        while (true) {
            if (!camera.read(frame)) {
                System.err.println("❌ Cannot read frame from camera");
                break;
            }

            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(gray, faces, 1.1, 5, 0,
                    new Size(30, 30), new Size());

            for (Rect face : faces.toArray()) {
                Imgproc.rectangle(frame, face.tl(), face.br(), new Scalar(0, 255, 0), 2);

                Mat faceROI = new Mat(gray, face);
                INDArray features = preprocessFace(faceROI);

                if (features.length() != model.layerInputSize(0)) {
                    System.err.println("⚠️ Skipping face: input size mismatch");
                    continue;
                }

                INDArray output = model.output(features);
                int predictedIdx = output.argMax(1).getInt(0);
                String name = indexToLabel.getOrDefault(predictedIdx, "Unknown");

                Imgproc.putText(frame, name, new Point((double) face.x, (double) (face.y - 10)),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 0, 0), 2);

                // TODO: Mark attendance logic (e.g., DB or file)
                System.out.println("✅ Attendance marked for: " + name);
            }

            HighGui.imshow("Face Attendance - Press 'q' to quit", frame);

            int key = HighGui.waitKey(30);
            if (key == 'q' || key == 27) {
                System.out.println("👋 Exiting attendance system.");
                break;
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
    }

    private INDArray preprocessFace(Mat face) {
        int rows = face.rows();
        int cols = face.cols();

        Imgproc.resize(face, face, new Size(100, 100)); // Ensuring consistent input size

        byte[] data = new byte[100 * 100];
        face.get(0, 0, data);

        float[] floatData = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            floatData[i] = (data[i] & 0xFF) / 255.0f;
        }

        return Nd4j.create(floatData).reshape(1, 10000);
    }

    public static void main(String[] args) throws Exception {
        // Load trained model
        String modelPath = "face_recognition_model.zip";
        MultiLayerNetwork model = MultiLayerNetwork.load(new File(modelPath), false);

        // Label to index map
        Map<String, Integer> labelToIndex = new HashMap<>();
        labelToIndex.put("user1", 0);
        labelToIndex.put("user2", 1);

        // Index to label map
        Map<Integer, String> indexToLabel = new HashMap<>();
        for (Entry<String, Integer> e : labelToIndex.entrySet()) {
            indexToLabel.put(e.getValue(), e.getKey());
        }

        // Haar cascade path
        URL cascadeUrl = FaceAttendance.class.getClassLoader().getResource("haarcascade_frontalface_default.xml");
        if (cascadeUrl == null) {
            throw new IllegalStateException("❌ Haar cascade file not found in resources folder.");
        }
        String haarCascadePath = new File(cascadeUrl.toURI()).getAbsolutePath();
        System.out.println("Loading Haar cascade from: " + haarCascadePath);

        FaceAttendance attendanceSystem = new FaceAttendance(haarCascadePath, model, indexToLabel);
        attendanceSystem.startAttendance();
    }
}
