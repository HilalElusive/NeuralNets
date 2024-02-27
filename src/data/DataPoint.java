package data;

public class DataPoint {
    public final double[] inputs;
    public final double[] expectedOutputs;
    public final int label;

    public DataPoint(double[] inputs, int label, int numLabels) {
        this.inputs = inputs;
        this.label = label;
        this.expectedOutputs = createOneHot(label, numLabels);
    }

    private static double[] createOneHot(int index, int num) {
        double[] oneHot = new double[num];
        if (index >= 0 && index < num) {
            oneHot[index] = 1.0;
        }
        return oneHot;
    }
}
