package functions;

public interface IActivation {
    double activate(double[] inputs, int index);

    double derivative(double[] inputs, int index);

    Activation.ActivationType getActivationType();
}
