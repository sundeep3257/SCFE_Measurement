####Downloads and Functions####

library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(scales)

find_southwick_angle <- function(x1, y1, x2, y2,
                                 x3, y3, x4, y4) {
  
  # Direction vector for Line A
  ax <- x2 - x1
  ay <- y2 - y1
  
  # Direction vector for the original second line
  bx_orig <- x4 - x3
  by_orig <- y4 - y3
  
  # Check for zero-length lines
  if (sqrt(ax^2 + ay^2) < 1e-10) {
    stop("Line A has zero length; angle cannot be calculated.")
  }
  
  if (sqrt(bx_orig^2 + by_orig^2) < 1e-10) {
    stop("Second line has zero length; perpendicular line cannot be calculated.")
  }
  
  # Direction vector for Line B, which is perpendicular to the second line
  # A perpendicular vector to (dx, dy) is (-dy, dx)
  bx <- -by_orig
  by <- bx_orig
  
  # Calculate angle between Line A and Line B using dot product
  dot_product <- ax * bx + ay * by
  
  mag_a <- sqrt(ax^2 + ay^2)
  mag_b <- sqrt(bx^2 + by^2)
  
  cos_theta <- dot_product / (mag_a * mag_b)
  
  # Clamp value to avoid numerical issues with acos()
  cos_theta <- max(min(cos_theta, 1), -1)
  
  angle_degrees <- acos(cos_theta) * 180 / pi
  
  # Return the smaller intersection angle
  smaller_angle <- min(angle_degrees, 180 - angle_degrees)
  
  return(smaller_angle)
}
find_angle_three_points <- function(x1, y1, x2, y2, x3, y3) {
  
  # Point 2 is the vertex
  v1 <- c(x1 - x2, y1 - y2)
  v2 <- c(x3 - x2, y3 - y2)
  
  mag1 <- sqrt(sum(v1^2))
  mag2 <- sqrt(sum(v2^2))
  
  if (mag1 < 1e-10 || mag2 < 1e-10) {
    stop("One of the points overlaps the vertex; angle cannot be calculated.")
  }
  
  cos_theta <- sum(v1 * v2) / (mag1 * mag2)
  
  # Prevent numerical issues with acos()
  cos_theta <- max(min(cos_theta, 1), -1)
  
  angle_degrees <- acos(cos_theta) * 180 / pi
  
  return(angle_degrees)
}
calculate_hn_offset <- function(xA, yA, xB, yB,
                                xC, yC, xD, yD,
                                diameter) {
  
  # Direction vector of line AB
  dx <- xB - xA
  dy <- yB - yA
  
  if (sqrt(dx^2 + dy^2) < 1e-10) {
    stop("Points A and B are identical; slope/direction cannot be calculated.")
  }
  
  if (diameter <= 0) {
    stop("Diameter must be greater than 0.")
  }
  
  # The two parallel lines pass through C and D and have direction vector (dx, dy).
  # Perpendicular distance between parallel lines:
  # distance = |dy*x - dx*y + b difference| / sqrt(dx^2 + dy^2)
  #
  # Equivalent point-line distance using C and D:
  perpendicular_distance <- abs(dy * (xD - xC) - dx * (yD - yC)) / sqrt(dx^2 + dy^2)
  
  hn_offset_ratio <- perpendicular_distance / diameter
  
  return(hn_offset_ratio)
}
calculate_psa <- function(A_x, A_y, B_x, B_y, C_x, C_y, D_x, D_y) {
  
  # Vector for line AB
  AB <- c(B_x - A_x, B_y - A_y)
  
  # Vector for line CD
  CD <- c(D_x - C_x, D_y - C_y)
  
  # Check for invalid points
  if (sqrt(sum(AB^2)) == 0) {
    stop("Points A and B cannot be identical.")
  }
  
  if (sqrt(sum(CD^2)) == 0) {
    stop("Points C and D cannot be identical.")
  }
  
  # A vector perpendicular to AB
  PERP <- c(-AB[2], AB[1])
  
  # Calculate angle between PERP and CD using dot product
  dot_product <- sum(PERP * CD)
  magnitude_product <- sqrt(sum(PERP^2)) * sqrt(sum(CD^2))
  
  angle_radians <- acos(dot_product / magnitude_product)
  angle_degrees <- angle_radians * 180 / pi
  
  # Convert to acute angle
  if (angle_degrees > 90) {
    angle_degrees <- 180 - angle_degrees
  }
  
  return(angle_degrees)
}
calculate_distance <- function(point1_x, point1_y, point2_x, point2_y) {
  distance <- sqrt((point2_x - point1_x)^2 + (point2_y - point1_y)^2)
  return(distance)
}


####Southwick Angle####

setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/01_Flipped_XR_Testing")
files <- list.files()

for (i in 1:length(files))
{
  #manual data
  {
    #get coords of shaft line
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/04_5Point_Detection/03_ShaftLine_Coords")
      file <- files[i]
      file <- gsub(".nii", ".csv", file, fixed = TRUE)
      manual_data <- read.csv(file, header = TRUE, sep = ",")
      start_x <- manual_data[1, 2]
      start_y <- manual_data[1, 3]
      end_x <- manual_data[2, 2]
      end_y <- manual_data[2, 3]
    }
    
    #get coords of physis tips
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/04_5Point_Detection/02_Manual_5Point_Coords")
      file <- files[i]
      file <- gsub(".nii", "_5points.csv", file, fixed = TRUE)
      manual_data <- read.csv(file, header = TRUE, sep = ",")
      med_edge_x <- manual_data[1, 2]
      med_edge_y <- manual_data[1, 3]
      lat_edge_x <- manual_data[2, 2]
      lat_edge_y <- manual_data[2, 3]
    }
    
    #calculate southwick angle
    {
      #find intersection
      manual_southwick <- find_southwick_angle(
        start_x, start_y,
        end_x, end_y,
        med_edge_x, med_edge_y,
        lat_edge_x, lat_edge_y
      )
      
    }
  }
  
  #pipeline data
  {
    #get coords of shaft line
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/05_ShaftLine_Coords")
      file <- files[i]
      file <- gsub(".nii", ".csv", file, fixed = TRUE)
      pipeline_data <- read.csv(file, header = TRUE, sep = ",")
      start_x <- pipeline_data[1, 2]
      start_y <- pipeline_data[1, 3]
      end_x <- pipeline_data[2, 2]
      end_y <- pipeline_data[2, 3]
    }
    
    #get coords of physis tips
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/04.1_PointPreds_CSV")
      file <- files[i]
      file <- gsub(".nii", "_predicted_5points.csv", file, fixed = TRUE)
      pipeline_data <- read.csv(file, header = TRUE, sep = ",")
      med_edge_x <- pipeline_data[1, 3]
      med_edge_y <- pipeline_data[1, 4]
      lat_edge_x <- pipeline_data[2, 3]
      lat_edge_y <- pipeline_data[2, 4]
    }
    
    #calculate southwick angle
    {
      #find intersection
      pipeline_southwick <- find_southwick_angle(
        start_x, start_y,
        end_x, end_y,
        med_edge_x, med_edge_y,
        lat_edge_x, lat_edge_y
      )
    }
  }
  
  file <- files[i]
  file <- gsub(".nii", "", file, fixed = TRUE)
  new_data <- data.frame(file, manual_southwick, pipeline_southwick)
  names(new_data) <- c("Filename", "Manual.Southwick", "Pipeline.Southwick")
  
  if (i == 1)
  {
    southwick_data <- new_data
  }
  if (i > 1)
  {
    southwick_data <- rbind(southwick_data, new_data)
  }
}


####Alpha Angle####

setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/01_Flipped_XR_Testing")
files <- list.files()

for (i in 1:length(files))
{
  #manual
  {
    #get distal shaft point
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/04_5Point_Detection/02_Manual_5Point_Coords")
      file <- files[i]
      file <- gsub(".nii", "_5points.csv", file, fixed = TRUE)
      manual_data <- read.csv(file, header = TRUE, sep = ",")
      distal_x <- manual_data[5, 2]
      distal_y <- manual_data[5, 3]
    }
    
    #get center of femur head circle and anterior point
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/04_5Point_Detection/04.1_HeadCircles_CSV")
      file <- files[i]
      file <- gsub(".nii", "_circle_measurements.csv", file, fixed = TRUE)
      manual_data <- read.csv(file, header = TRUE, sep = ",")
      circle_center_x <- manual_data[1, 2]
      circle_center_y <- manual_data[1, 3]
      anterior_point_x <- manual_data[1, 6]
      anterior_point_y <- manual_data[1, 7]
    }
    
    manual_alpha <- find_angle_three_points(distal_x, distal_y,
                                            circle_center_x, circle_center_y,
                                            anterior_point_x, anterior_point_y)
  }
  
  #pipeline
  {
    #get distal shaft point
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/04.1_PointPreds_CSV")
      file <- files[i]
      file <- gsub(".nii", "_predicted_5points.csv", file, fixed = TRUE)
      pipeline_data <- read.csv(file, header = TRUE, sep = ",")
      distal_x <- pipeline_data[5, 3]
      distal_y <- pipeline_data[5, 4]
    }
    
    #get center of femur head circle and anterior point
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/06.1_HeadCircles_CSV")
      file <- files[i]
      file <- gsub(".nii", "_circle_measurements.csv", file, fixed = TRUE)
      pipeline_data <- read.csv(file, header = TRUE, sep = ",")
      circle_center_x <- pipeline_data[1, 2]
      circle_center_y <- pipeline_data[1, 3]
      anterior_point_x <- pipeline_data[1, 6]
      anterior_point_y <- pipeline_data[1, 7]
    }
    
    pipeline_alpha <- find_angle_three_points(distal_x, distal_y,
                                            circle_center_x, circle_center_y,
                                            anterior_point_x, anterior_point_y)
  }
  
  file <- files[i]
  file <- gsub(".nii", "", file, fixed = TRUE)
  new_data <- data.frame(file, manual_alpha, pipeline_alpha)
  names(new_data) <- c("Filename", "Manual.Alpha", "Pipeline.Alpha")
  
  if (i == 1)
  {
    alpha_data <- new_data
  }
  if (i > 1)
  {
    alpha_data <- rbind(alpha_data, new_data)
  }
}


####Head-Neck Offset Ratio####

setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/01_Flipped_XR_Testing")
files <- list.files()

for (i in 1:length(files))
{
  #manual data
  {
    #get coords of neck line
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/03_Femur_Neck_Line/02_NeckLine_Manual_Coords")
      file <- files[i]
      file <- gsub(".nii", "_line.csv", file, fixed = TRUE)
      manual_data <- read.csv(file, header = TRUE, sep = ",")
      start_x <- manual_data[1, 2]
      start_y <- manual_data[1, 3]
      end_x <- manual_data[2, 2]
      end_y <- manual_data[2, 3]
    }
    
    #get coords of anterior head point and neck divot
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/04_5Point_Detection/02_Manual_5Point_Coords")
      file <- files[i]
      file <- gsub(".nii", "_5points.csv", file, fixed = TRUE)
      manual_data <- read.csv(file, header = TRUE, sep = ",")
      anthead_x <- manual_data[3, 2]
      anthead_y <- manual_data[3, 3]
      neckdivot_x <- manual_data[4, 2]
      neckdivot_y <- manual_data[4, 3]
    }
    
    #get diameter of approximated head
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/04_5Point_Detection/04.1_HeadCircles_CSV")
      file <- files[i]
      file <- gsub(".nii", "_circle_measurements.csv", file, fixed = TRUE)
      manual_data <- read.csv(file, header = TRUE, sep = ",")
      diameter <- manual_data[1, 5]
    }
    
    manual_hno <- calculate_hn_offset(start_x, start_y, end_x, end_y,
                                      anthead_x, anthead_y, neckdivot_x, neckdivot_y,
                                      diameter)
  }
  
  #pipeline data
  {
    #get coords of neck line
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/03.1_NeckLine_CSV")
      file <- files[i]
      file <- gsub(".nii", ".csv", file, fixed = TRUE)
      pipeline_data <- read.csv(file, header = TRUE, sep = ",")
      start_x <- pipeline_data[1, 2]
      start_y <- pipeline_data[1, 3]
      end_x <- pipeline_data[2, 2]
      end_y <- pipeline_data[2, 3]
    }
    
    #get coords of anterior head point and neck divot
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/04.1_PointPreds_CSV")
      file <- files[i]
      file <- gsub(".nii", "_predicted_5points.csv", file, fixed = TRUE)
      pipeline_data <- read.csv(file, header = TRUE, sep = ",")
      anthead_x <- pipeline_data[3, 3]
      anthead_y <- pipeline_data[3, 4]
      neckdivot_x <- pipeline_data[4, 3]
      neckdivot_y <- pipeline_data[4, 4]
    }
    
    #get diameter of approximated head
    {
      setwd("C:/Users/sunde/Box/SCFE Measurement/05_Pipeline_Testing/06.1_HeadCircles_CSV")
      file <- files[i]
      file <- gsub(".nii", "_circle_measurements.csv", file, fixed = TRUE)
      pipeline_data <- read.csv(file, header = TRUE, sep = ",")
      diameter <- pipeline_data[1, 5]
    }
    
    pipeline_hno <- calculate_hn_offset(start_x, start_y, end_x, end_y,
                                      anthead_x, anthead_y, neckdivot_x, neckdivot_y,
                                      diameter)
  }
  
  file <- files[i]
  file <- gsub(".nii", "", file, fixed = TRUE)
  new_data <- data.frame(file, manual_hno, pipeline_hno)
  names(new_data) <- c("Filename", "Manual.HNO", "Pipeline.HNO")
  
  if (i == 1)
  {
    hno_data <- new_data
  }
  if (i > 1)
  {
    hno_data <- rbind(hno_data, new_data)
  }
}