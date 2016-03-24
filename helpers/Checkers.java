package helpers;

public class Checkers {
	public static boolean isValid(String model) {
		return model.equalsIgnoreCase("socBIT") || model.equalsIgnoreCase("soRec") ;	// model.equalsIgnoreCase("STE") || model.equalsIgnoreCase("bSTE")
	}
}
