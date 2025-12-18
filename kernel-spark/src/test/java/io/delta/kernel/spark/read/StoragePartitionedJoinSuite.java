/*
 * Copyright (2025) The Delta Lake Project Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.delta.kernel.spark.read;

import static org.junit.jupiter.api.Assertions.*;

import io.delta.kernel.spark.SparkDsv2TestBase;
import io.delta.kernel.spark.catalog.SparkTable;
import java.io.File;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.read.partitioning.KeyGroupedPartitioning;
import org.apache.spark.sql.connector.read.partitioning.Partitioning;
import org.apache.spark.sql.connector.read.partitioning.UnknownPartitioning;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.execution.exchange.Exchange;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import scala.collection.JavaConverters;

/**
 * Test suite for Storage Partitioned Join (SPJ) functionality.
 *
 * <p>SPJ enables shuffle-free joins for tables with matching partition columns.
 */
public class StoragePartitionedJoinSuiteNew extends SparkDsv2TestBase {

  @Test
  public void testPartitionedTableReportsKeyGroupedPartitioning(@TempDir File tempDir) {
    String path = tempDir.getAbsolutePath();
    spark.sql(
        String.format(
            "CREATE TABLE test_tbl (id INT, name STRING) USING delta LOCATION '%s' PARTITIONED BY"
                + " (name)",
            path));
    spark.sql("INSERT INTO test_tbl VALUES (1, 'a'), (2, 'b'), (3, 'a')");

    SparkScan scan = createScan(path);
    Partitioning partitioning = scan.outputPartitioning();

    assertTrue(partitioning instanceof KeyGroupedPartitioning);
    KeyGroupedPartitioning kgp = (KeyGroupedPartitioning) partitioning;
    assertEquals(1, kgp.keys().length);
    assertEquals("name", kgp.keys()[0].toString());
    assertEquals(2, kgp.numPartitions()); // 'a' and 'b'
  }

  @Test
  public void testNonPartitionedTableReportsUnknownPartitioning(@TempDir File tempDir) {
    String path = tempDir.getAbsolutePath();
    spark.sql(
        String.format(
            "CREATE TABLE test_tbl (id INT, name STRING) USING delta LOCATION '%s'", path));
    spark.sql("INSERT INTO test_tbl VALUES (1, 'a'), (2, 'b')");

    SparkScan scan = createScan(path);
    Partitioning partitioning = scan.outputPartitioning();

    assertTrue(partitioning instanceof UnknownPartitioning);
    assertEquals(0, ((UnknownPartitioning) partitioning).numPartitions());
  }

  @Test
  public void testMultiColumnPartitioning(@TempDir File tempDir) {
    String path = tempDir.getAbsolutePath();
    spark.sql(
        String.format(
            "CREATE TABLE test_tbl (id INT, yr INT, mo INT) USING delta LOCATION '%s' PARTITIONED"
                + " BY (yr, mo)",
            path));
    spark.sql("INSERT INTO test_tbl VALUES (1, 2023, 1), (2, 2023, 2), (3, 2024, 1)");

    SparkScan scan = createScan(path);
    Partitioning partitioning = scan.outputPartitioning();

    assertTrue(partitioning instanceof KeyGroupedPartitioning);
    KeyGroupedPartitioning kgp = (KeyGroupedPartitioning) partitioning;
    assertEquals(2, kgp.keys().length);
    assertEquals(3, kgp.numPartitions()); // (2023,1), (2023,2), (2024,1)
  }

  @Test
  public void testJoinOnMatchingPartitions(@TempDir File dir1, @TempDir File dir2) {
    // Create two tables partitioned by the same column
    String path1 = dir1.getAbsolutePath();
    String path2 = dir2.getAbsolutePath();

    spark.sql(
        String.format(
            "CREATE TABLE t1 (id INT, cat STRING, val DOUBLE) USING delta LOCATION '%s' PARTITIONED"
                + " BY (cat)",
            path1));
    spark.sql("INSERT INTO t1 VALUES (1, 'A', 10.0), (2, 'B', 20.0), (3, 'A', 30.0)");

    spark.sql(
        String.format(
            "CREATE TABLE t2 (id INT, cat STRING, price DOUBLE) USING delta LOCATION '%s'"
                + " PARTITIONED BY (cat)",
            path2));
    spark.sql("INSERT INTO t2 VALUES (1, 'A', 100.0), (2, 'B', 200.0), (4, 'A', 400.0)");

    Dataset<Row> result =
        spark.sql("SELECT t1.id, t1.cat, t1.val, t2.price FROM t1 JOIN t2 ON t1.cat = t2.cat");

    assertEquals(
        5, result.count()); // (1,A) + (3,A) join with (1,A) + (4,A) = 4, (2,B) join (2,B) = 1
  }

  @Test
  public void testJoinOnNonPartitionedTables(@TempDir File dir1, @TempDir File dir2) {
    String path1 = dir1.getAbsolutePath();
    String path2 = dir2.getAbsolutePath();

    spark.sql(
        String.format("CREATE TABLE t1 (id INT, name STRING) USING delta LOCATION '%s'", path1));
    spark.sql("INSERT INTO t1 VALUES (1, 'Alice'), (2, 'Bob')");

    spark.sql(
        String.format("CREATE TABLE t2 (id INT, val DOUBLE) USING delta LOCATION '%s'", path2));
    spark.sql("INSERT INTO t2 VALUES (1, 10.0), (2, 20.0)");

    Dataset<Row> result =
        spark.sql("SELECT t1.id, t1.name, t2.val FROM t1 JOIN t2 ON t1.id = t2.id");

    assertEquals(2, result.count());
  }

  @Test
  public void testJoinWithDifferentPartitionColumns(@TempDir File dir1, @TempDir File dir2) {
    // t1 partitioned by cat, t2 partitioned by region -> no SPJ benefit
    String path1 = dir1.getAbsolutePath();
    String path2 = dir2.getAbsolutePath();

    spark.sql(
        String.format(
            "CREATE TABLE t1 (id INT, cat STRING) USING delta LOCATION '%s' PARTITIONED BY (cat)",
            path1));
    spark.sql("INSERT INTO t1 VALUES (1, 'A'), (2, 'B')");

    spark.sql(
        String.format(
            "CREATE TABLE t2 (id INT, region STRING) USING delta LOCATION '%s' PARTITIONED BY"
                + " (region)",
            path2));
    spark.sql("INSERT INTO t2 VALUES (1, 'US'), (2, 'EU')");

    Dataset<Row> result =
        spark.sql("SELECT t1.id, t1.cat, t2.region FROM t1 JOIN t2 ON t1.id = t2.id");

    assertEquals(2, result.count());
  }

  @Test
  public void testJoinWithSupersetPartitions(@TempDir File dir1, @TempDir File dir2) {
    // t1 partitioned by (yr, mo), t2 partitioned by (yr) -> t1 is superset
    String path1 = dir1.getAbsolutePath();
    String path2 = dir2.getAbsolutePath();

    spark.sql(
        String.format(
            "CREATE TABLE t1 (id INT, yr INT, mo INT) USING delta LOCATION '%s' PARTITIONED BY (yr,"
                + " mo)",
            path1));
    spark.sql("INSERT INTO t1 VALUES (1, 2023, 1), (2, 2023, 2), (3, 2024, 1)");

    spark.sql(
        String.format(
            "CREATE TABLE t2 (id INT, yr INT) USING delta LOCATION '%s' PARTITIONED BY (yr)",
            path2));
    spark.sql("INSERT INTO t2 VALUES (1, 2023), (3, 2024)");

    Dataset<Row> result = spark.sql("SELECT t1.id, t1.yr, t1.mo FROM t1 JOIN t2 ON t1.yr = t2.yr");

    assertEquals(3, result.count()); // All t1 rows match (2023 matches 2, 2024 matches 1)
  }

  @Test
  public void testJoinWithSubsetPartitions(@TempDir File dir1, @TempDir File dir2) {
    // t1 partitioned by (yr), t2 partitioned by (yr, mo) -> t2 is superset
    String path1 = dir1.getAbsolutePath();
    String path2 = dir2.getAbsolutePath();

    spark.sql(
        String.format(
            "CREATE TABLE t1 (id INT, yr INT) USING delta LOCATION '%s' PARTITIONED BY (yr)",
            path1));
    spark.sql("INSERT INTO t1 VALUES (1, 2023), (3, 2024)");

    spark.sql(
        String.format(
            "CREATE TABLE t2 (id INT, yr INT, mo INT) USING delta LOCATION '%s' PARTITIONED BY (yr,"
                + " mo)",
            path2));
    spark.sql("INSERT INTO t2 VALUES (1, 2023, 1), (2, 2023, 2), (3, 2024, 1)");

    Dataset<Row> result = spark.sql("SELECT t2.id, t2.yr, t2.mo FROM t1 JOIN t2 ON t1.yr = t2.yr");

    assertEquals(3, result.count()); // All t2 rows match
  }

  @Test
  public void testSPJDisabled(@TempDir File tempDir) {
    String path = tempDir.getAbsolutePath();
    spark.sql(
        String.format(
            "CREATE TABLE test_tbl (id INT, cat STRING) USING delta LOCATION '%s' PARTITIONED BY"
                + " (cat)",
            path));
    spark.sql("INSERT INTO test_tbl VALUES (1, 'A'), (2, 'B')");

    // Create scan with SPJ disabled
    java.util.HashMap<String, String> opts = new java.util.HashMap<>();
    opts.put("enableStoragePartitionedJoin", "false");
    SparkTable table =
        new SparkTable(
            Identifier.of(new String[] {"spark_catalog", "default"}, "test_tbl"),
            path,
            new CaseInsensitiveStringMap(opts));

    ScanBuilder builder = table.newScanBuilder(new CaseInsensitiveStringMap(opts));
    SparkScan scan = (SparkScan) builder.build();
    Partitioning partitioning = scan.outputPartitioning();

    assertTrue(partitioning instanceof UnknownPartitioning);
  }

  private SparkScan createScan(String path) {
    CaseInsensitiveStringMap options = new CaseInsensitiveStringMap(new java.util.HashMap<>());
    SparkTable table =
        new SparkTable(
            Identifier.of(new String[] {"spark_catalog", "default"}, "test_tbl"), path, options);
    ScanBuilder builder = table.newScanBuilder(options);
    return (SparkScan) builder.build();
  }

  private boolean containsExchange(SparkPlan plan) {
    if (plan instanceof Exchange) {
      return true;
    }
    return JavaConverters.seqAsJavaList(plan.children()).stream().anyMatch(this::containsExchange);
  }
}
