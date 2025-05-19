package com.attendance;

import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.highgui.HighGui;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.BorderLayout;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.HashMap;
import java.util.Map;

public class FaceAttendanceApp extends JFrame {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private JLabel videoLabel;
    private JTextArea logArea;
    private JButton toggleButton;
    private volatile boolean running = false;

    private CascadeClassifier faceDetector;
    private MultiLayerNetwork model;
    private Map<Integer, String> indexToLabel;

    public FaceAttendanceApp(String cascadePath,
                             MultiLayerNetwork model,
                             Map<Integer, String> indexToLabel) {
        this.faceDetector = new CascadeClassifier(cascadePath);
        if (faceDetector.empty())
            throw new IllegalStateException("Cannot load cascade: " + cascadePath);

        this.model = model;
        this.indexToLabel = indexToLabel;

        setTitle("Face Attendance");
        setLayout(new BorderLayout());
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        videoLabel = new JLabel();
        add(videoLabel, BorderLayout.CENTER);

        logArea = new JTextArea(10, 20);
        logArea.setEditable(false);
        add(new JScrollPane(logArea), BorderLayout.EAST);

        toggleButton = new JButton("Start Attendance");
        toggleButton.addActionListener(e -> {
            if (!running) startCapture();
            else stopCapture();
        });
        add(toggleButton, BorderLayout.SOUTH);

        setSize(900, 600);
        setVisible(true);
    }

    private void startCapture() {
        running = true;
        toggleButton.setText("Stop Attendance");
        new Thread(this::captureLoop).start();
    }

    private void stopCapture() {
        running = false;
        toggleButton.setText("Start Attendance");
    }

    private void captureLoop() {
        VideoCapture cam = new VideoCapture(0);
        if (!cam.isOpened()) {
            log("❌ Cannot open camera");
            return;
        }

        Mat frame = new Mat();
        Map<String, Boolean> marked = new HashMap<>();
        SimpleDateFormat fmt = new SimpleDateFormat("HH:mm:ss");

        while (running) {
            if (!cam.read(frame)) break;

            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(gray, faces, 1.1, 5, 0,
                    new Size(30, 30), new Size());

            for (Rect r : faces.toArray()) {
                Imgproc.rectangle(frame, r.tl(), r.br(), new Scalar(0, 255, 0), 2);

                Mat roi = new Mat(gray, r);
                INDArray input = preprocess(roi);
                INDArray output = model.output(input);
                int idx = output.argMax(1).getInt(0);
                String name = indexToLabel.getOrDefault(idx, "Unknown");

                Imgproc.putText(frame,
                                name,
                                new org.opencv.core.Point(r.x, r.y - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                new Scalar(255, 0, 0),
                                2);

                if (!marked.containsKey(name)) {
                    marked.put(name, true);
                    log("✅ " + name + " at " + fmt.format(new java.util.Date()));
                }
            }

            SwingUtilities.invokeLater(() ->
                videoLabel.setIcon(new ImageIcon(matToBufferedImage(frame)))
            );

            try { Thread.sleep(33); } catch (InterruptedException ignored) {}
        }

        cam.release();
    }

    private INDArray preprocess(Mat face) {
        int W = 100, H = 100;
        Mat resized = new Mat();
        Imgproc.resize(face, resized, new Size(W, H));

        byte[] data = new byte[W * H];
        resized.get(0, 0, data);

        float[] fdata = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            fdata[i] = (data[i] & 0xFF) / 255f;
        }
        return Nd4j.create(fdata).reshape(1, W * H);
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = mat.channels() > 1 ?
                BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;
        BufferedImage img = new BufferedImage(mat.cols(), mat.rows(), type);
        byte[] b = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        mat.get(0, 0, b);
        return img;
    }

    private void log(String msg) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(msg + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }

    public static void main(String[] args) throws Exception {
        // Load model
        MultiLayerNetwork model = MultiLayerNetwork.load(
            new File("face_recognition_model.zip"), false);

        // Label maps
        Map<String, Integer> l2i = new HashMap<>();
        l2i.put("user1", 0);
        l2i.put("user2", 1);
        Map<Integer, String> i2l = new HashMap<>();
        for (Map.Entry<String, Integer> e : l2i.entrySet()) {
            i2l.put(e.getValue(), e.getKey());
        }

        // Cascade from resources
        URL url = FaceAttendanceApp.class.getClassLoader()
                          .getResource("haarcascade_frontalface_default.xml");
        if (url == null) throw new IllegalStateException("Cascade XML not found");
        String cascadePath = new File(url.toURI()).getAbsolutePath();

        SwingUtilities.invokeLater(() ->
            new FaceAttendanceApp(cascadePath, model, i2l)
        );
    }
}
