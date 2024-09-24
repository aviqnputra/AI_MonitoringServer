import os
import time
from ultralytics import YOLO
import cv2
import psycopg2

def send_to_database(person_count):
    # Connect to your PostgreSQL database
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="1",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
        
        # Insert the prediction data into a different table
    query = "INSERT INTO dataorang_client1 (orang) VALUES (%s)"
    cursor.execute(query, (person_count,))
        
    conn.commit()
    cursor.close()
    conn.close()
        
        
        
def process_images(model, input_folder, output_folder, annotation_folder, image_folder):
    # Keep track of processed images
    processed_images = set()

    try:
        while True:
            # Process each image in the input folder
            for filename in os.listdir(input_folder):
                if filename in processed_images:
                    continue  # Skip already processed images
                
                # Read the image
                image_path = os.path.join(input_folder, filename)
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Error: Could not read image {filename}.")
                    continue

                # Perform detection
                results = model(frame)

                # Draw bounding boxes and count people
                person_count = 0
                annotations = []
                for result in results:
                    for box in result.boxes:
                        if box.cls == 0 and box.conf > 0.8:  # class 0 is 'person' in COCO dataset and confidence > 0.8
                            person_count += 1
                            # Get the bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract individual tensor elements
                            # Convert confidence score to float
                            confidence_score = float(box.conf)
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Put label
                            cv2.putText(frame, f'Person {confidence_score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                            # Create annotation in YOLO format
                            x1, y1, x2, y2 = box.xyxy[0]
                            width = x2 - x1
                            height = y2 - y1
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2

                            # Normalize the coordinates
                            img_width, img_height = 640, 480  # Replace with your image width and height
                            x_center /= img_width
                            y_center /= img_height
                            width /= img_width
                            height /= img_height

                            annotation = f"{int(box.cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                            
                            annotations.append(annotation)

                # If no person detected, skip this image
                if person_count == 0:
                    send_to_database(person_count)
                    continue

                # Add total person count to the image
                cv2.putText(frame, f'Total Persons: {person_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Save the original image without bounding boxes and class labels
                original_image_path = os.path.join(image_folder, filename)
                cv2.imwrite(original_image_path, cv2.imread(image_path))

                # Save the resulting frame with bounding boxes and person count
                output_path = os.path.join(output_folder, f"result_{filename}")
                cv2.imwrite(output_path, frame)

                # Save the annotations in YOLO format
                annotation_path = os.path.join(annotation_folder, f"{os.path.splitext(filename)[0]}.txt")
                with open(annotation_path, 'w') as f:
                    f.write('\n'.join(annotations))

                # Print the person count for each image
                print(f"Image {filename}: Person Count = {person_count}")

                # Send person count to the database
                send_to_database(person_count)

                # Mark this image as processed
                processed_images.add(filename)
            
            # Wait for a short time before checking the folder again
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopped")

def main():
    input_folder = "/root/projectAI/Yolo/Input"
    output_folder = "/root/projectAI/Yolo/Output"
    annotation_folder = "/root/projectAI/Yolo/Data/labels"
    image_folder = "/root/projectAI/Yolo/Data/images"

    # Load YOLOv8 model
    model = YOLO('/root/projectAI/Yolo/best.pt')  # Ensure the YOLOv8 model is downloaded and in the correct path
    process_images(model, input_folder, output_folder, annotation_folder, image_folder)

if __name__ == "__main__":
    main()
