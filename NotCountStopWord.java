package com;

/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class NotCountStopWord {

  public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    private ArrayList<String> StopWords = new ArrayList<String>(Arrays.asList("ourselves", "hers", "between",
        "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own",
        "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is",
        "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we",
        "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down",
        "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at",
        "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then",
        "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself",
        "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being",
        "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"));

    private String clean(String s) {
      String temp = "";
      for (int i = 0; i < s.length(); i++) {
        if (Character.isDigit(s.charAt(i)) || Character.isLetter(s.charAt(i))) {
          temp += s.charAt(i);
        }
      }
      return temp;
    }

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      JSONParser parser = new JSONParser();
      try {
        JSONObject json = (JSONObject) parser.parse(value.toString());
        StringTokenizer itr = new StringTokenizer(json.get("text").toString());
        while (itr.hasMoreTokens()) {
          String wordS = clean(itr.nextToken());
          if (!StopWords.contains(wordS.toLowerCase())) {
            word.set(wordS);
            context.write(word, one);
          }
        }
      } catch (ParseException e) {
        e.printStackTrace();
      }

    }
  }

  public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length < 2) {
      System.err.println("Usage: wordcount <in> [<in>...] <out>");
      System.exit(2);
    }
    Job job = new Job(conf, "word count");
    job.setJarByClass(NotCountStopWord.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    for (int i = 0; i < otherArgs.length - 1; ++i) {
      FileInputFormat.addInputPath(job, new Path(otherArgs[i]));
    }
    FileOutputFormat.setOutputPath(job, new Path(otherArgs[otherArgs.length - 1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}