package kaggle.sentiment;

import org.tartarus.snowball.ext.PorterStemmer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;


public class ReviewCleaner {

	/**
	 * Returns `text` with all sub-strings matching the regular expression `regex`
	 * replaced by the string `normalizationString`
	 */
	public static String applyNormalizationTemplate(String text, String regex, String normalizationString) {
		Pattern pattern = Pattern.compile(regex);
		Matcher matcher = pattern.matcher(text);

		return matcher.replaceAll(normalizationString);
	}

	/**
	 * Returns `text` with every occurrence of one of the currency symbol
	 * '$', '€' or '£' replaced by the string literal " normalizedcurrencysymbol ".
	 */
	public static String normalizeCurrencySymbol(String text) {
		String regex = "[\\$\\€\\£]";
		return applyNormalizationTemplate(text, regex, " normalizedcurrencysymbol ");
	}

	/**
	 * Returns `text` with every occurrence of one of a number
	 * replaced by the string literal "normalizednumber".
	 */
	public static String normalizeNumbers(String text) {
		String regex = "\\d+";
		return applyNormalizationTemplate(text, regex, " normalizednumber ");
	}

	/**
	 * Returns `text` with every occurrence of one of an URL
	 * replaced by the string literal " normalizedurl ".
	 */
	public static String normalizeURL(String text) {
		String regex = "\\b((?:[a-z][\\w-]+:(?:/{1,3}|[a-z0-9%])|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\))+(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]{};:'\".,<>?«»“”‘’]))";
		return applyNormalizationTemplate(text, regex, " normalizedurl ");
	}

	public static String removeNotLetters(String text) {
		return applyNormalizationTemplate(text, "[^a-zA-Z]", " ");
	}

	public static List<String> nGram(List<String> words, int n) {
		return IntStream.range(0, words.size() - n + 1)
				.mapToObj(i -> new ArrayList<>(words.subList(i, i + n)).stream().collect(joining(" ")))
				.collect(toList());
	}

	public static String[] cleanReview(String text) {
		return Optional.of(text.toLowerCase())
				.map(ReviewCleaner::normalizeCurrencySymbol)
				.map(ReviewCleaner::normalizeNumbers)
				.map(ReviewCleaner::normalizeURL)
				.map(ReviewCleaner::removeNotLetters)
				.get().trim()
				.split("\\s+");

	}

	public static List<String> reviewToTokens(String text) {
		List<String> uniGrams = stemWords(cleanReview(text));
		List<String> biGrams = nGram(uniGrams, 2);
		biGrams.addAll(uniGrams);
		return biGrams;
	}

	public static List<String> stemWords(String[] uniGrams) {
		PorterStemmer stemmer = new PorterStemmer();
		return Arrays.stream(uniGrams).map(s -> {
			stemmer.setCurrent(s);
			stemmer.stem();
			return stemmer.getCurrent();
		}).collect(toList());
	}


}
