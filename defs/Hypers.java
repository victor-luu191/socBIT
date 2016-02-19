package defs;

public class Hypers {
	
	public double topicLambda; 
	public double brandLambda;
	public double weightLambda;
	public double decisionLambda;
	
	public double alpha;	// tuning param only for STE, alpha controls how much a user believe in his own ratings vs. friend's ratings 
	
	public Hypers(double topicLambda, double alpha) {
		this.topicLambda = topicLambda;
		this.alpha = alpha;
	}

	public Hypers(double topicLambda, double brandLambda, double weightLambda, double decisionLambda) {
		
		this.topicLambda = topicLambda;
		this.brandLambda = brandLambda;
		this.weightLambda = weightLambda;
		this.decisionLambda = decisionLambda;
	}

	public static Hypers setBySocBIT(double topicLambda, double brandLambda, double weightLambda, double decisionLambda) {
		return new Hypers(topicLambda, brandLambda, weightLambda, decisionLambda);
	}
	
	public static Hypers setBySTE(double topicLambda, double alpha) {
		return new Hypers(topicLambda, alpha);
	}
}
