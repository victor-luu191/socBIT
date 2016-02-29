package defs;

public class NonConvergeException extends Exception {

	private static final long serialVersionUID = 1L;
	public NonConvergeException() {
		String msg = "Not converged yet but already exceeded the maximum number of iterations. Gradient descent stopped!!!";
		System.out.println(msg);
	}
}
