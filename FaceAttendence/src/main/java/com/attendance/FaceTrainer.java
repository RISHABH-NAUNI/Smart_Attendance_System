package com.attendance;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.*;

public class FaceTrainer {

    private static final int IMG_WIDTH = 100;   // fixed width
    private static final int IMG_HEIGHT = 100;  // fixed height

    private Map<String, Integer> labelToIndex = new HashMap<>();
    private List<DataSet> dataSetList = new ArrayList<>();

    // Load images, convert to DataSet objects
    public void loadTrainingData(String datasetPath) {
        File baseDir = new File(datasetPath);
        if (!baseDir.exists() || !baseDir.isDirectory()) {
            System.err.println("Invalid dataset directory: " + datasetPath);
            return;
        }

        int classIndex = 0;
        // Map labels to indices
        for (File userFolder : Objects.requireNonNull(baseDir.listFiles())) {
            if (userFolder.isDirectory()) {
                String label = userFolder.getName();
                labelToIndex.put(label, classIndex++);
            }
        }

        int numClasses = labelToIndex.size();

        for (File userFolder : Objects.requireNonNull(baseDir.listFiles())) {
            if (!userFolder.isDirectory()) continue;
            String label = userFolder.getName();
            int labelIdx = labelToIndex.get(label);

            for (File imgFile : Objects.requireNonNull(userFolder.listFiles())) {
                if (!(imgFile.getName().endsWith(".png") || imgFile.getName().endsWith(".jpg"))) continue;

                if (!imgFile.exists() || !imgFile.canRead()) {
                    System.err.println("⚠️ File does not exist or cannot be read: " + imgFile.getAbsolutePath());
                    continue;
                }

                Mat img = Imgcodecs.imread(imgFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
                if (img.empty()) {
                    System.err.println("⚠️ Failed to load image (empty Mat): " + imgFile.getAbsolutePath());
                    continue;
                }

                // Resize to fixed size
                Mat resizedImg = new Mat();
                Imgproc.resize(img, resizedImg, new org.opencv.core.Size(IMG_WIDTH, IMG_HEIGHT));

                INDArray features = matToINDArray(resizedImg);
                INDArray labels = Nd4j.zeros(numClasses);
                labels.putScalar(labelIdx, 1.0);

                DataSet dataSet = new DataSet(features, labels);
                dataSetList.add(dataSet);
            }
        }

        System.out.println("Loaded training data for " + numClasses + " classes, total samples: " + dataSetList.size());
    }

    // Convert OpenCV Mat to flattened INDArray normalized between 0-1
    private INDArray matToINDArray(Mat mat) {
        int rows = mat.rows();
        int cols = mat.cols();
        byte[] data = new byte[rows * cols];
        mat.get(0, 0, data);

        float[] floatData = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            floatData[i] = (data[i] & 0xFF) / 255.0f;
        }
        return Nd4j.create(floatData).reshape(1, rows * cols); // shape: [1, rows*cols]
    }

    public void trainModel() {
        if (dataSetList.isEmpty()) {
            System.err.println("No training data loaded");
            return;
        }

        int inputSize = (int) dataSetList.get(0).getFeatures().length();
        int outputSize = labelToIndex.size();

        // Stack all features and labels vertically
        INDArray allFeatures = Nd4j.vstack(dataSetList.stream().map(DataSet::getFeatures).toArray(INDArray[]::new));
        INDArray allLabels = Nd4j.vstack(dataSetList.stream().map(DataSet::getLabels).toArray(INDArray[]::new));

        DataSet fullDataSet = new DataSet(allFeatures, allLabels);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        System.out.println("Training started...");

        for (int i = 0; i < 20; i++) {
            model.fit(fullDataSet);
            System.out.println("Epoch " + (i + 1) + " completed");
        }

        try {
            model.save(new File("face_recognition_model.zip"));
            System.out.println("Model saved as face_recognition_model.zip");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public Map<String, Integer> getLabelToIndexMap() {
        return labelToIndex;
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        FaceTrainer trainer = new FaceTrainer();
        trainer.loadTrainingData("dataset");  // Set your dataset folder path here
        trainer.trainModel();
    }
}
