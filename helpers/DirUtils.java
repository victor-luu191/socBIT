package helpers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DirUtils {
	public static void mkDir(String name) throws IOException {
		if (!Files.exists(Paths.get(name))) {
			Files.createDirectories(Paths.get(name));
		}
	}
}
