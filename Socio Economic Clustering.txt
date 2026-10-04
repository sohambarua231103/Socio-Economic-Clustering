# ============================================================
# SOCIO-ECONOMIC COUNTRY SEGMENTATION
# K-MEANS CLUSTERING + PCA IN R
# ============================================================

# ============================================================
# 1. INSTALL AND LOAD REQUIRED PACKAGES
# ============================================================

packages <- c(
  "ggplot2",
  "dplyr",
  "factoextra",
  "cluster",
  "readr"
)

installed <- rownames(installed.packages())

for (p in packages) {
  if (!(p %in% installed)) {
    install.packages(p)
  }
}

library(ggplot2)
library(dplyr)
library(factoextra)
library(cluster)
library(readr)


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Life Expectancy / socio-economic dataset
# Contains socio-economic and health indicators for countries

data <- read.csv("LifeExpectancyData.csv")

# Display first observations
head(data)

# Dataset dimensions
dim(data)

# Dataset structure
str(data)

# Summary statistics
summary(data)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

# Remove leading/trailing spaces from column names
names(data) <- trimws(names(data))

# Display column names
print(names(data))


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Check missing values
missing_values <- colSums(is.na(data))

print(missing_values)

# Remove rows with missing values
data_clean <- na.omit(data)

# Check dimensions after cleaning
cat("\nOriginal observations:", nrow(data))
cat("\nObservations after removing missing values:",
    nrow(data_clean), "\n")

# ============================================================
# 5. SELECT RELEVANT VARIABLES
# ============================================================

features <- c(
  "Life.expectancy",
  "Adult.Mortality",
  "infant.deaths",
  "Alcohol",
  "percentage.expenditure",
  "Hepatitis.B",
  "BMI",
  "under.five.deaths",
  "Polio",
  "Total.expenditure",
  "Diphtheria",
  "HIV.AIDS",
  "GDP",
  "Population",
  "thinness..1.19.years",
  "thinness.5.9.years",
  "Income.composition.of.resources",
  "Schooling"
)

# Check which columns are missing
missing_features <- setdiff(
  features,
  names(data_clean)
)

if (length(missing_features) > 0) {

  cat("\nThe following columns were not found:\n")
  print(missing_features)

} else {

  cluster_data <- data_clean[, features]

  cat(
    "\nAll features found successfully.\n"
  )
}

# Display selected column names
print(names(cluster_data))

# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

# Summary statistics
summary(cluster_data)

# Check missing values again
colSums(is.na(cluster_data))


# ============================================================
# 7. DISTRIBUTION OF LIFE EXPECTANCY
# ============================================================

ggplot(
  cluster_data,
  aes(x = Life.expectancy)
) +
  geom_histogram(
    bins = 30
  ) +
  labs(
    title = "Distribution of Life Expectancy",
    x = "Life Expectancy",
    y = "Number of Observations"
  ) +
  theme_minimal()


# ============================================================
# 8. GDP DISTRIBUTION
# ============================================================

ggplot(
  cluster_data,
  aes(x = GDP)
) +
  geom_histogram(
    bins = 30
  ) +
  labs(
    title = "Distribution of GDP",
    x = "GDP",
    y = "Number of Observations"
  ) +
  theme_minimal()


# ============================================================
# 9. CORRELATION MATRIX
# ============================================================

correlation_matrix <- cor(
  cluster_data,
  use = "complete.obs"
)

print(round(correlation_matrix, 2))


# ============================================================
# 10. STANDARDIZE FEATURES
# ============================================================

# Standardization is important because variables
# have different scales.

scaled_data <- scale(cluster_data)

# Convert to data frame
scaled_data <- as.data.frame(scaled_data)

# Check standardized data
head(scaled_data)


# ============================================================
# 11. FIND OPTIMAL NUMBER OF CLUSTERS
# USING ELBOW METHOD
# ============================================================

set.seed(123)

fviz_nbclust(
  scaled_data,
  kmeans,
  method = "wss",
  k.max = 10
) +
  labs(
    title = "Elbow Method for Optimal Number of Clusters"
  )


# ============================================================
# 12. SILHOUETTE METHOD
# ============================================================

fviz_nbclust(
  scaled_data,
  kmeans,
  method = "silhouette",
  k.max = 10
) +
  labs(
    title = "Silhouette Method for Optimal Number of Clusters"
  )


# ============================================================
# 13. SET NUMBER OF CLUSTERS
# ============================================================

# Based on the Elbow and Silhouette analysis,
# choose the appropriate number of clusters.

k <- 3


# ============================================================
# 14. K-MEANS CLUSTERING
# ============================================================

set.seed(123)

kmeans_model <- kmeans(
  scaled_data,
  centers = k,
  nstart = 25,
  iter.max = 100
)

# Display K-means results
print(kmeans_model)


# ============================================================
# 15. CLUSTER ASSIGNMENTS
# ============================================================

cluster_data$Cluster <- factor(
  kmeans_model$cluster
)

# Display cluster sizes
table(cluster_data$Cluster)


# ============================================================
# 16. CLUSTER VISUALIZATION
# ============================================================

fviz_cluster(
  kmeans_model,
  data = scaled_data,
  geom = "point",
  ellipse.type = "convex",
  main = "K-Means Country Segmentation"
)


# ============================================================
# 17. PCA - PRINCIPAL COMPONENT ANALYSIS
# ============================================================

pca_model <- prcomp(
  scaled_data,
  center = TRUE,
  scale. = FALSE
)

# PCA summary
summary(pca_model)


# ============================================================
# 18. PCA VARIANCE EXPLAINED
# ============================================================

variance_explained <- pca_model$sdev^2 /
  sum(pca_model$sdev^2)

pca_variance <- data.frame(
  Component = paste0(
    "PC",
    1:length(variance_explained)
  ),
  Variance = variance_explained * 100
)

print(pca_variance)


# ============================================================
# 19. PCA SCREE PLOT
# ============================================================

ggplot(
  pca_variance[1:10, ],
  aes(
    x = Component,
    y = Variance
  )
) +
  geom_col() +
  labs(
    title = "PCA Explained Variance",
    x = "Principal Component",
    y = "Variance Explained (%)"
  ) +
  theme_minimal()


# ============================================================
# 20. PCA DATASET
# ============================================================

pca_data <- as.data.frame(
  pca_model$x
)

pca_data$Cluster <- cluster_data$Cluster


# ============================================================
# 21. PCA CLUSTER VISUALIZATION
# ============================================================

ggplot(
  pca_data,
  aes(
    x = PC1,
    y = PC2,
    shape = Cluster
  )
) +
  geom_point(size = 3) +
  labs(
    title = "Country Clusters using PCA",
    x = "Principal Component 1",
    y = "Principal Component 2"
  ) +
  theme_minimal()


# ============================================================
# 22. PCA BIPLOT
# ============================================================

fviz_pca_biplot(
  pca_model,
  repel = TRUE,
  title = "PCA Biplot of Socio-Economic Indicators"
)

# ============================================================
# 23. CLUSTER PROFILING
# ============================================================

cluster_profile <- cluster_data %>%
  group_by(Cluster) %>%
  summarise(
    Countries = n(),

    Average_Life_Expectancy =
      mean(Life.expectancy, na.rm = TRUE),

    Average_Adult_Mortality =
      mean(Adult.Mortality, na.rm = TRUE),

    Average_Infant_Deaths =
      mean(infant.deaths, na.rm = TRUE),

    Average_GDP =
      mean(GDP, na.rm = TRUE),

    Average_BMI =
      mean(BMI, na.rm = TRUE),

    Average_HIV =
      mean(HIV.AIDS, na.rm = TRUE),

    Average_Schooling =
      mean(Schooling, na.rm = TRUE),

    Average_Income_Index =
      mean(
        Income.composition.of.resources,
        na.rm = TRUE
      )
  )

print(cluster_profile)
# ============================================================
# 24. STANDARDIZED CLUSTER PROFILE
# ============================================================

cluster_means <- aggregate(
  scaled_data,
  by = list(Cluster = kmeans_model$cluster),
  FUN = mean
)

print(cluster_means)


# ============================================================
# 25. CLUSTER SIZE VISUALIZATION
# ============================================================

cluster_sizes <- as.data.frame(
  table(cluster_data$Cluster)
)

colnames(cluster_sizes) <- c(
  "Cluster",
  "Count"
)

ggplot(
  cluster_sizes,
  aes(
    x = Cluster,
    y = Count
  )
) +
  geom_col() +
  labs(
    title = "Number of Countries in Each Cluster",
    x = "Cluster",
    y = "Number of Countries"
  ) +
  theme_minimal()


# ============================================================
# 26. LIFE EXPECTANCY BY CLUSTER
# ============================================================

ggplot(
  cluster_data,
  aes(
    x = Cluster,
    y = Life.expectancy
  )
) +
  geom_boxplot() +
  labs(
    title = "Life Expectancy Across Clusters",
    x = "Cluster",
    y = "Life Expectancy"
  ) +
  theme_minimal()


# ============================================================
# 27. GDP BY CLUSTER
# ============================================================

ggplot(
  cluster_data,
  aes(
    x = Cluster,
    y = GDP
  )
) +
  geom_boxplot() +
  labs(
    title = "GDP Across Clusters",
    x = "Cluster",
    y = "GDP"
  ) +
  theme_minimal()


# ============================================================
# 28. SCHOOLING BY CLUSTER
# ============================================================

ggplot(
  cluster_data,
  aes(
    x = Cluster,
    y = Schooling
  )
) +
  geom_boxplot() +
  labs(
    title = "Schooling Across Clusters",
    x = "Cluster",
    y = "Average Schooling"
  ) +
  theme_minimal()


# ============================================================
# 29. SILHOUETTE ANALYSIS
# ============================================================

distance_matrix <- dist(
  scaled_data
)

silhouette_result <- silhouette(
  kmeans_model$cluster,
  distance_matrix
)

# Average silhouette width
average_silhouette <- mean(
  silhouette_result[, 3]
)

cat(
  "\nAverage Silhouette Width:",
  round(average_silhouette, 4),
  "\n"
)


# ============================================================
# 30. SILHOUETTE PLOT
# ============================================================

fviz_silhouette(
  silhouette_result
) +
  labs(
    title = "Silhouette Analysis of K-Means Clusters"
  )


# ============================================================
# 31. CLUSTER CENTERS
# ============================================================

cluster_centers <- as.data.frame(
  kmeans_model$centers
)

print(cluster_centers)


# ============================================================
# 32. IDENTIFY COUNTRY CLUSTERS
# ============================================================

country_names <- data_clean$Country

country_clusters <- data.frame(
  Country = country_names,
  Cluster = kmeans_model$cluster
)

# Sort by cluster
country_clusters <- country_clusters[
  order(country_clusters$Cluster),
]

print(head(country_clusters, 30))


# ============================================================
# 33. DISPLAY COUNTRIES BY CLUSTER
# ============================================================

for (i in 1:k) {

  cat(
    "\n====================================\n"
  )

  cat(
    "CLUSTER",
    i,
    "\n"
  )

  cat(
    "====================================\n"
  )

  print(
    country_clusters$Country[
      country_clusters$Cluster == i
    ]
  )
}


# ============================================================
# 34. CREATE FINAL DATASET
# ============================================================

final_data <- data_clean

final_data$Cluster <- kmeans_model$cluster

final_data$Cluster <- factor(
  final_data$Cluster
)

# Display final dataset
head(final_data)


# ============================================================
# 35. SAVE CLUSTERED DATA
# ============================================================

write.csv(
  final_data,
  "country_clustered_dataset.csv",
  row.names = FALSE
)


# ============================================================
# 36. SAVE CLUSTER PROFILE
# ============================================================

write.csv(
  cluster_profile,
  "cluster_profile.csv",
  row.names = FALSE
)


# ============================================================
# 37. SAVE PCA RESULTS
# ============================================================

write.csv(
  pca_data,
  "PCA_cluster_results.csv",
  row.names = FALSE
)


# ============================================================
# 38. SAVE COUNTRY CLUSTERS
# ============================================================

write.csv(
  country_clusters,
  "country_cluster_assignments.csv",
  row.names = FALSE
)


# ============================================================
# 39. FINAL PROJECT SUMMARY
# ============================================================

cat("\n")
cat("============================================================\n")
cat("     SOCIO-ECONOMIC COUNTRY SEGMENTATION PROJECT\n")
cat("============================================================\n")

cat(
  "\nNumber of countries/observations:",
  nrow(cluster_data)
)

cat(
  "\nNumber of clusters:",
  k
)

cat(
  "\nAverage silhouette width:",
  round(average_silhouette, 4)
)

cat("\n\nCluster sizes:\n")

print(
  table(cluster_data$Cluster)
)

cat("\n============================================================\n")
cat("              PROJECT COMPLETED\n")
cat("============================================================\n")