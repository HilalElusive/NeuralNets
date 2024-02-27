package functions;

public class Activation {

    public enum ActivationType {
        Sigmoid, TanH, ReLU, Softmax, LeakyReLU
    }

    public static IActivation getActivationFromType(ActivationType type) {
        switch (type) {
            case Sigmoid:
                return new Sigmoid();
            case TanH:
                return new TanH();
            case ReLU:
                return new ReLU();
            case LeakyReLU:
                return new ReLU();
            case Softmax:
                return new Softmax();
            default:
                throw new IllegalArgumentException("Unhandled activation type");
        }
    }

    public static class Sigmoid implements IActivation {
        public double activate(double[] inputs, int index) {
            return 1.0 / (1 + Math.exp(-inputs[index]));
        }

        public double derivative(double[] inputs, int index) {
            double a = activate(inputs, index);
            return a * (1 - a);
        }

        public ActivationType getActivationType() {
            return ActivationType.Sigmoid;
        }
    }

    public static class TanH implements IActivation {
        public double activate(double[] inputs, int index) {
            double e2 = Math.exp(2 * inputs[index]);
            return (e2 - 1) / (e2 + 1);
        }

        public double derivative(double[] inputs, int index) {
            double e2 = Math.exp(2 * inputs[index]);
            double t = (e2 - 1) / (e2 + 1);
            return 1 - t * t;
        }

        public ActivationType getActivationType() {
            return ActivationType.TanH;
        }
    }

    public static class ReLU implements IActivation {
        public double activate(double[] inputs, int index) {
            return Math.max(0, inputs[index]);
        }

        public double derivative(double[] inputs, int index) {
            return (inputs[index] > 0) ? 1 : 0;
        }

        public ActivationType getActivationType() {
            return ActivationType.ReLU;
        }
    }

    public static class LeakyReLU implements IActivation {
        private double alpha = 0.01; // This is the usual value, but you can experiment with it

        public double activate(double[] inputs, int index) {
            return inputs[index] > 0 ? inputs[index] : alpha * inputs[index];
        }

        public double derivative(double[] inputs, int index) {
            return inputs[index] > 0 ? 1 : alpha;
        }

        public ActivationType getActivationType() {
            return ActivationType.LeakyReLU;
        }
    }
    
    public static class Softmax implements IActivation {
        public double activate(double[] inputs, int index) {
            double expSum = 0;
            for (int i = 0; i < inputs.length; i++) {
                expSum += Math.exp(inputs[i]);
            }

            return Math.exp(inputs[index]) / expSum;
        }

        public double derivative(double[] inputs, int index) {
            double expSum = 0;
            for (int i = 0; i < inputs.length; i++) {
                expSum += Math.exp(inputs[i]);
            }

            double ex = Math.exp(inputs[index]);
            return (ex * expSum - ex * ex) / (expSum * expSum);
        }

        public ActivationType getActivationType() {
            return ActivationType.Softmax;
        }
    }
}