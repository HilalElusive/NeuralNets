package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSVDataLoader {

    public static DataPoint[] readCSV(String filePath) {
        List<DataPoint> dataPoints = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                if (values.length == 785) { // 1 label + 784 features
                    int label = Integer.parseInt(values[0]);
                    double[] features = new double[784];
                    for (int i = 0; i < 784; i++) {
                        features[i] = Double.parseDouble(values[i + 1]) / 255; //Normalization
                    }
                    // MNIST has 10 different labels (0-9)
                    DataPoint dataPoint = new DataPoint(features, label, 10);
                    dataPoints.add(dataPoint);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataPoints.toArray(new DataPoint[0]);
    }
}