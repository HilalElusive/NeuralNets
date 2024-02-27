package functions;

public class Cost {
	public enum CostType {
        MeanSquareError, CrossEntropy
    }

    public static ICost getCostFromType(CostType type) {
        switch (type) {
            case MeanSquareError:
                return new MeanSquaredError();
            case CrossEntropy:
                return new CrossEntropy();
            default:
                throw new IllegalArgumentException("Unhandled cost type");
        }
    }

    public static class MeanSquaredError implements ICost {
        public double costFunction(double[] predictedOutputs, double[] expectedOutputs) {
            double cost = 0;
            for (int i = 0; i < predictedOutputs.length; i++) {
                double error = predictedOutputs[i] - expectedOutputs[i];
                cost += error * error;
            }
            return 0.5 * cost;
        }

        public double costDerivative(double predictedOutput, double expectedOutput) {
            return predictedOutput - expectedOutput;
        }

        public CostType costFunctionType() {
            return CostType.MeanSquareError;
        }
    }

    public static class CrossEntropy implements ICost {
        public double costFunction(double[] predictedOutputs, double[] expectedOutputs) {
            double cost = 0;
            for (int i = 0; i < predictedOutputs.length; i++) {
                double x = predictedOutputs[i];
                double y = expectedOutputs[i];
                double v = (y == 1) ? -Math.log(x) : -Math.log(1 - x);
                cost += Double.isNaN(v) ? 0 : v;
            }
            return cost;
        }

        public double costDerivative(double predictedOutput, double expectedOutput) {
            double x = predictedOutput;
            double y = expectedOutput;
            if (x == 0 || x == 1) {
                return 0;
            }
            return (-x + y) / (x * (x - 1));
        }

        public CostType costFunctionType() {
            return CostType.CrossEntropy;
        }
    }
}
