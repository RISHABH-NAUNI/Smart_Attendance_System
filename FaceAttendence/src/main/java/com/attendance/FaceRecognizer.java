package com.attendance;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.Map;

public class FaceRecognizer {

    private MultiLayerNetwork model;
    private Map<Integer, String> indexToLabel;

    public FaceRecognizer(MultiLayerNetwork model, Map<String, Integer> labelToIndex) {
        this.model = model;

        // Reverse labelToIndex for prediction lookup
        indexToLabel = new java.util.HashMap<>();
        for (Map.Entry<String, Integer> entry : labelToIndex.entrySet()) {
            indexToLabel.put(entry.getValue(), entry.getKey());
        }
    }

    // Converts Mat (28x28 grayscale) to INDArray normalized to [0,1]
    private INDArray matToINDArray(Mat mat) {
        int rows = mat.rows();
        int cols = mat.cols();
        byte[] data = new byte[rows * cols];
        mat.get(0, 0, data);

        float[] floatData = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            floatData[i] = (data[i] & 0xFF) / 255.0f;
        }
        return Nd4j.create(floatData).reshape(1, rows * cols);
    }

    // Recognize face from a Mat image, returns predicted label
    public String recognize(Mat faceImage) {
        Mat grayFace = new Mat();
        if (faceImage.channels() > 1) {
            Imgproc.cvtColor(faceImage, grayFace, Imgproc.COLOR_BGR2GRAY);
        } else {
            grayFace = faceImage.clone();
        }

        Mat resizedFace = new Mat();
        Imgproc.resize(grayFace, resizedFace, new Size(28, 28)); // resize to model input size

        INDArray input = matToINDArray(resizedFace);

        INDArray output = model.output(input);
        int predictedIdx = Nd4j.argMax(output, 1).getInt(0);

        return indexToLabel.getOrDefault(predictedIdx, "Unknown");
    }
}
