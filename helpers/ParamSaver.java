package helpers;

import java.io.IOException;
import java.util.Arrays;

import myUtil.Savers;
import defs.Params;
import defs.SocBIT_Params;

public class ParamSaver {
	
	public static void save(Params params, String dir) throws IOException {
		
		if (params instanceof SocBIT_Params) {
			saveTopicComponents(params, dir);
			saveBrandComponents(params, dir);
		}
		else {
			saveTopicComponents(params, dir);
		}
	}

	private static void saveBrandComponents(Params params, String dir) throws IOException {
		SocBIT_Params castParams = (SocBIT_Params) params;
		String userBrandFeat_file = dir + "user_brand_feats.csv";
		Savers.save(castParams.brandUser.toString(), userBrandFeat_file);
		
		String itemBrandFeat_file = dir + "item_brand_feats.csv";
		Savers.save(castParams.brandItem.toString(), itemBrandFeat_file);
		
		String decisionPref_file = dir + "decision_prefs.csv";
		Savers.save(Arrays.toString(castParams.userDecisionPrefs), decisionPref_file);
	}

	private static void saveTopicComponents(Params params, String dir) throws IOException {
		String userTopicFeat_file = dir + "user_topic_feats.csv";
		Savers.save(params.topicUser.toString(), userTopicFeat_file);
		
		String itemTopicFeat_file = dir + "item_topic_feats.csv";
		Savers.save(params.topicItem.toString(), itemTopicFeat_file);
	}
}
