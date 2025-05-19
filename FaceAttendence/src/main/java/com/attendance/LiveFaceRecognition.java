package com.attendance;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class LiveFaceRecognition extends JPanel {

    private static final long serialVersionUID = 1L;
    private VideoCapture camera;
    private CascadeClassifier faceDetector;
    private MultiLayerNetwork model;
    private Map<Integer, String> indexToLabel;

    private Mat frame = new Mat();
    private BufferedImage image;

    public LiveFaceRecognition(MultiLayerNetwork model, Map<String, Integer> labelToIndex, String haarCascadePath) {
        this.model = model;
        this.faceDetector = new CascadeClassifier(haarCascadePath);
        this.camera = new VideoCapture(0);

        // Reverse the label map for easy lookup
        this.indexToLabel = new HashMap<>();
        for (Map.Entry<String, Integer> entry : labelToIndex.entrySet()) {
            indexToLabel.put(entry.getValue(), entry.getKey());
        }

        if (!camera.isOpened()) {
            System.out.println("Error: Camera not detected!");
            System.exit(1);
        }

        JFrame frame = new JFrame("Live Face Recognition");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.add(this);
        frame.setVisible(true);

        // Start the camera capture in a new thread
        new Thread(this::captureLoop).start();
    }

    private void captureLoop() {
        // Use OpenCV's Point explicitly to avoid ambiguity
        org.opencv.core.Point textPos = new org.opencv.core.Point();

        while (true) {
            if (camera.read(frame)) {
                Mat gray = new Mat();
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

                MatOfRect faces = new MatOfRect();
                faceDetector.detectMultiScale(gray, faces);

                for (Rect rect : faces.toArray()) {
                    // Draw rectangle around detected face
                    Imgproc.rectangle(frame, rect, new Scalar(0, 255, 0));

                    // Extract face ROI and resize to 28x28 pixels (model input size)
                    Mat face = new Mat(gray, rect);
                    Mat resizedFace = new Mat();
                    Imgproc.resize(face, resizedFace, new Size(28, 28));

                    // Convert face Mat to INDArray normalized between 0 and 1
                    INDArray features = matToINDArray(resizedFace);

                    // Predict label using DL4J model
                    INDArray output = model.output(features);
                    int predictedIdx = Nd4j.argMax(output, 1).getInt(0);
                    String label = indexToLabel.getOrDefault(predictedIdx, "Unknown");

                    // Set text position just above face rectangle, avoid negative y
                    textPos.x = rect.x;
                    textPos.y = Math.max(rect.y - 10, 15);

                    // Put label text above the face rectangle
                    Imgproc.putText(frame, label, textPos,
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
                }

                // Convert OpenCV Mat to BufferedImage for Swing display
                image = matToBufferedImage(frame);
                repaint();  // repaint JPanel to update image
            }
        }
    }

    private INDArray matToINDArray(Mat mat) {
        int rows = mat.rows();
        int cols = mat.cols();
        byte[] data = new byte[rows * cols];
        mat.get(0, 0, data);

        float[] floatData = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            floatData[i] = (data[i] & 0xFF) / 255.0f;
        }
        return Nd4j.create(floatData).reshape(1, rows * cols);  // Shape: [1, features]
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] b = new byte[bufferSize];
        mat.get(0, 0, b);

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (image != null) {
            // Resize image to fit panel size
            g.drawImage(image, 0, 0, getWidth(), getHeight(), null);
        }
    }

    public static void main(String[] args) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the trained DL4J model
        File modelFile = new File("face_recognition_model.zip");
        MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, false);

        // Label map same as used during training
        Map<String, Integer> labelToIndex = new HashMap<>();
        labelToIndex.put("PersonA", 0);
        labelToIndex.put("PersonB", 1);
        // Add your own labels here...

        // Path to your Haar cascade XML file
        String haarCascadePath = "haarcascade_frontalface_default.xml";

        new LiveFaceRecognition(model, labelToIndex, haarCascadePath);
    }
}
