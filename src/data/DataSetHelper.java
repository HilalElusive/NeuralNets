package data;

import java.util.Random;

public class DataSetHelper {

    // Custom class to hold training and validation DataPoint arrays
    public static class SplitDataSet {
        public DataPoint[] train;
        public DataPoint[] validate;

        public SplitDataSet(DataPoint[] train, DataPoint[] validate) {
            this.train = train;
            this.validate = validate;
        }
    }

    // Split data into training and validation sets
    public static SplitDataSet splitData(DataPoint[] allData, float trainingSplit, boolean shuffle) {
        if (shuffle) {
            shuffleArray(allData, new Random());
        }

        int trainCount = (int) (allData.length * Math.min(trainingSplit, 1.0f));
        int validationCount = allData.length - trainCount;

        DataPoint[] trainData = new DataPoint[trainCount];
        DataPoint[] validationData = new DataPoint[validationCount];

        System.arraycopy(allData, 0, trainData, 0, trainCount);
        System.arraycopy(allData, trainCount, validationData, 0, validationCount);

        return new SplitDataSet(trainData, validationData);
    }

    public static Batch[] createMiniBatches(DataPoint[] allData, int size, boolean shuffle) {
        if (shuffle) {
            shuffleArray(allData, new Random());
        }

        int numBatches = (int) Math.ceil((double) allData.length / size);
        Batch[] batches = new Batch[numBatches];
        for (int i = 0; i < numBatches; i++) {
            int batchSize = Math.min(size, allData.length - i * size);
            DataPoint[] batchData = new DataPoint[batchSize];
            System.arraycopy(allData, i * size, batchData, 0, batchSize);
            batches[i] = new Batch(batchData);
        }
        return batches;
    }

    public static void shuffleBatches(Batch[] batches) {
        shuffleArray(batches, new Random());
    }

    private static <T> void shuffleArray(T[] array, Random prng) {
        for (int i = array.length - 1; i > 0; i--) {
            int index = prng.nextInt(i + 1);
            // Simple swap
            T a = array[index];
            array[index] = array[i];
            array[i] = a;
        }
    }
}
