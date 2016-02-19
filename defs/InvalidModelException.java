package defs;

public class InvalidModelException extends Exception {
	
	private static final long serialVersionUID = 1L;

	public InvalidModelException() {
		String msg = "Invalid model. Current implementation only accepts either socBIT or STE model. "
				+ "Pls type or copy the correct model name next time.";
		
		System.out.println(msg);
	}
	
	public InvalidModelException(String msg) {
		super(msg);
	}
}
