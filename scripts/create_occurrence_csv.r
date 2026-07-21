#!/usr/bin/env Rscript --vanilla

# Generate a CSV that can be imported into an occurrence database.
# First run download_occurrence_data, and reference its output path below.

# Load required libraries
library(ebirdst)
library(terra)

# File paths
ebird_data_path <- "/my/species_data"     # output path in download_occurrence_data
species_list_file <- "/my/species.csv"    # CSV with a 'species' column
county_list_file <- "/my/county_list.csv" # CSV with columns 'county_code', 'lat_min', 'lat_max', 'lon_min', 'lon_max'

# Load species and county data
species_list <- read.csv(species_list_file)$species
county_list <- read.csv(county_list_file)

# Convert eBird's 52 dated weekly layers to HawkEars' canonical 48 bins:
# four bins per month covering days 1-7, 8-14, 15-21, and 22-end. A
# month's fifth eBird week is averaged into the fourth bin. This is the
# same date-to-bin mapping used by HawkEars during inference.
parse_week_dates <- function(layer_names) {
  parse_one <- function(layer_name) {
    # terra may make syntactic names such as X2023.01.04; normalize those
    # as well as the YYYY-MM-DD and MM-DD names used by ebirdst releases.
    normalized <- sub("^X(?=[0-9])", "", layer_name, perl = TRUE)
    normalized <- gsub("\\.", "-", normalized)

    if (grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", normalized)) {
      date <- as.Date(normalized, format = "%Y-%m-%d")
    } else if (grepl("^[0-9]{2}-[0-9]{2}$", normalized)) {
      # Only month and day affect the HawkEars bin. Use a non-leap
      # reference year when older products omit the year.
      date <- as.Date(paste0("2001-", normalized), format = "%Y-%m-%d")
    } else {
      date <- as.Date(NA)
    }

    if (is.na(date)) NA_character_ else format(date, "%Y-%m-%d")
  }

  dates <- as.Date(vapply(layer_names, parse_one, character(1)))
  if (anyNA(dates)) {
    invalid_names <- paste(layer_names[is.na(dates)], collapse = ", ")
    stop("Unable to parse eBird raster layer dates: ", invalid_names)
  }
  dates
}

hawkears_bins <- function(dates) {
  month <- as.integer(format(dates, "%m"))
  day <- as.integer(format(dates, "%d"))
  (month - 1L) * 4L + pmin(3L, (day - 1L) %/% 7L)
}

# Function to process a single species and county
process_species_county <- function(species, county_code, lat_min, lat_max, lon_min, lon_max) {
  # Load the occurrence raster
  occurrence_raster <- load_raster(species, product = c("occurrence"), path = ebird_data_path)
  week_dates <- parse_week_dates(names(occurrence_raster))
  bins <- hawkears_bins(week_dates)

  expected_bins <- 0:47
  missing_bins <- setdiff(expected_bins, unique(bins))
  unexpected_bins <- setdiff(unique(bins), expected_bins)
  if (length(missing_bins) > 0 || length(unexpected_bins) > 0) {
    stop(
      "Invalid eBird-to-HawkEars bin mapping for ", species,
      "; missing bins: ", paste(missing_bins, collapse = ", "),
      "; unexpected bins: ", paste(unexpected_bins, collapse = ", ")
    )
  }

  # Define county extent and project to raster CRS
  county_extent <- vect(ext(lon_min, lon_max, lat_min, lat_max), crs = "EPSG:4326")
  county_proj <- project(county_extent, crs(occurrence_raster))

  # Crop raster to county extent
  cropped_raster <- crop(occurrence_raster, county_proj)

  # Calculate mean occurrence probability
  mean_occurrence <- global(cropped_raster, "mean", na.rm = TRUE)
  weekly_values <- mean_occurrence$mean
  bin_values <- vapply(
    expected_bins,
    function(bin) mean(weekly_values[bins == bin], na.rm = TRUE),
    numeric(1)
  )
  source_dates <- vapply(
    expected_bins,
    function(bin) paste(format(week_dates[bins == bin], "%Y-%m-%d"), collapse = ";"),
    character(1)
  )

  if (length(bin_values) != 48) {
    stop("Expected 48 HawkEars occurrence bins for ", species)
  }

  if (all(bin_values == 0, na.rm = TRUE)) {
    # Return an empty data frame if no data for this species
    return(data.frame())
  } else {
    # Return a data frame with results
    return(data.frame(
      species = species,
      county = county_code,
      occurrence_bin = expected_bins,
      source_dates = source_dates,
      mean_occurrence = bin_values
    ))
  }
}

# Loop over each county and species
for (i in 1:nrow(county_list)) {
  # Initialize an empty data frame to store results
  county_results <- data.frame()

  county_code <- county_list$county_code[i]
  lat_min <- county_list$lat_min[i]
  lat_max <- county_list$lat_max[i]
  lon_min <- county_list$lon_min[i]
  lon_max <- county_list$lon_max[i]

  for (species in species_list) {
    cat("Processing species:", species, "for county:", county_code, "\n")

    # Process the species and county
    result <- process_species_county(
      species = species,
      county_code = county_code,
      lat_min = lat_min,
      lat_max = lat_max,
      lon_min = lon_min,
      lon_max = lon_max
    )

    # Append results
    county_results <- rbind(county_results, result)
  }
  # Save the results for this county to a CSV file
  output_file <- paste0("/my/output_dir/", paste0(county_code, ".csv"))
  write.csv(county_results, output_file, row.names = FALSE)
  cat("Results for county", county_code, "saved to", output_file, "\n")
}
cat("All counties processed. Individual CSV files created for each county.\n")
