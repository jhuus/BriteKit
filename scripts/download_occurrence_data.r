# Use ebirdst to download occurrence data for a list of species

library(data.table)
library(ebirdst)

# path to a CSV with a single "species" column
species_list_file <- "/my/species.csv"

# read the CSV file into a data frame
species_list <- read.csv(species_list_file)$species

# process values from the first column, skipping unavailable species (ebirdst omits many species)
unavailable <- c("xxx", "yyy")
output_path = "/my/species_data"
for (species in species_list) {
  if (!(species %in% unavailable)) {
    print(species)
    ebirdst_download_status(species, download_abundance = FALSE, download_occurrence = TRUE,
                            download_pis = FALSE, pattern = "_occurrence_median_3km", force = FALSE,
                            path = output_path)
  }
}
