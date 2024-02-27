package functions;

public interface ICost {
    double costFunction(double[] predictedOutputs, double[] expectedOutputs);

    double costDerivative(double predictedOutput, double expectedOutput);

    Cost.CostType costFunctionType();
}