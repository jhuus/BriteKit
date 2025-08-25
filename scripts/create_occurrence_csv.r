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

# Function to process a single species and county
process_species_county <- function(species, county_code, lat_min, lat_max, lon_min, lon_max) {
  # Load the occurrence raster
  occurrence_raster <- load_raster(species, product = c("occurrence"), path = ebird_data_path)

  # Define county extent and project to raster CRS
  county_extent <- vect(ext(lon_min, lon_max, lat_min, lat_max), crs = "EPSG:4326")
  county_proj <- project(county_extent, crs(occurrence_raster))

  # Crop raster to county extent
  cropped_raster <- crop(occurrence_raster, county_proj)

  # Calculate mean occurrence probability
  mean_occurrence <- global(cropped_raster, "mean", na.rm = TRUE)
  if (all(mean_occurrence == 0, na.rm = TRUE)) {
    # Return an empty data frame if no data for this species
    return(data.frame())
  } else {
    # Return a data frame with results
    return(data.frame(
      species = species,
      county = county_code,
      mean_occurrence = mean_occurrence$mean
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
