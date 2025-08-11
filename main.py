import json
import cv2
from bisect import bisect_left
import pandas as pd

class Person:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.entered = False
        self.exited = False
        self.entry_time = None
        self.exit_time = None
        self.positions = []

    @property
    def center(self):
        """Возвращает текущий центр bounding box'а"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update_position(self, new_bbox):
        """Обновляет позицию человека и сохраняет историю"""
        self.bbox = new_bbox
        self.positions.append(self.center)

    def has_crossed_line(self, line_start, line_end, direction):
        """
        Проверяет, пересек ли человек линию в заданном направлении
        direction: 1 - выход, -1 - вход
        """
        if len(self.positions) < 2:
            return False

        prev_center = self.positions[-2]
        current_center = self.positions[-1]

        cross = line_crossed(prev_center, current_center, line_start, line_end)
        return cross == direction


def line_crossed(prev_center, current_center, line_start, line_end):
    """
    Определяет, пересекает ли отрезок (prev_center, current_center) линию (line_start, line_end)
    с использованием векторного произведения для определения пересечения.
    Возвращает:
    1 - пересечение слева направо (выход)
    -1 - пересечение справа налево (вход)
    0 - нет пересечения
    """

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    A = prev_center
    B = current_center
    C = line_start
    D = line_end

    ccw1 = ccw(A, C, D)
    ccw2 = ccw(B, C, D)
    ccw3 = ccw(A, B, C)
    ccw4 = ccw(A, B, D)

    if ((ccw1 * ccw2 < 0) and (ccw3 * ccw4 < 0)):
        line_vec = (D[0] - C[0], D[1] - C[1])
        move_vec = (B[0] - A[0], B[1] - A[1])
        cross = line_vec[0] * move_vec[1] - line_vec[1] * move_vec[0]

        if cross < 0:
            return 1  # выход
        else:
            return -1  # вход

    return 0


def process_frame_data(fr, people, int_line_coords, ext_line_coords, current_datetime, enter_count, exit_count,
                       current_frame_people):
    for person_data in fr["detected"]["person"]:
        x1, y1 = int(person_data[0]), int(person_data[1])
        x2, y2 = int(person_data[2]), int(person_data[3])
        bbox = [x1, y1, x2, y2]
        track_id = None
        if len(person_data) > 5 and isinstance(person_data[5], dict):
            for key in person_data[5].keys():
                if 'track_id' in person_data[5][key]:
                    track_id = person_data[5][key]['track_id']
                    break
        if track_id is not None:
            current_frame_people.add(track_id)
            if track_id not in people:
                people[track_id] = Person(track_id, bbox)
            else:
                people[track_id].update_position(bbox)

            person = people[track_id]

            if not person.exited and person.has_crossed_line(
                    (ext_line_coords[0], ext_line_coords[1]),
                    (ext_line_coords[2], ext_line_coords[3]),
                    1
            ):
                person.exited = True
                person.exit_time = current_datetime
                exit_count += 1
                print(f"Person {track_id} exited at {person.exit_time}")

            if not person.entered and person.has_crossed_line(
                    (int_line_coords[0], int_line_coords[1]),
                    (int_line_coords[2], int_line_coords[3]),
                    -1
            ):
                person.entered = True
                person.entry_time = current_datetime
                enter_count += 1
                print(f"Person {track_id} entered at {person.entry_time}")

    return enter_count, exit_count


def main():
    mode = input("Выберите режим ('play' для воспроизведения видео, 'count-only' для подсчета без видео): ").strip().lower()
    while mode not in ['play', 'count-only']:
        print("Неверный режим. Пожалуйста, выберите 'play' или 'count-only'.")
        mode = input("Выберите режим ('play' для воспроизведения видео, 'count-only' для подсчета без видео): ").strip().lower()

    enter_count = 0
    exit_count = 0
    people_data = []

    data = json.load(open("detections.json"))["eventSpecific"]["nnDetect"]["10_8_3_203_rtsp_camera_3"]
    frames = data["frames"]
    config = data["cfg"]
    timestamps = sorted([float(fr["timestamp"]) for fr in frames.values()])
    frames_dict = {fr["timestamp"]: fr for fr in frames.values()}

    box = config["cross_lines"][0]["box"]
    video_frames = config["video_frames"]
    int_line = config["cross_lines"][0]["int_line"]
    exit_line = config["cross_lines"][0]["ext_line"]

    def validate_coords(x, y):
        return x / box[0] * video_frames["frame_width"], y / box[0] * video_frames["frame_width"]

    int_1 = validate_coords(int_line[0], int_line[1])
    int_2 = validate_coords(int_line[2], int_line[3])
    ext_1 = validate_coords(exit_line[0], exit_line[1])
    ext_2 = validate_coords(exit_line[2], exit_line[3])
    int_line_coords = (int(int_1[0]), int(int_1[1]), int(int_2[0]), int(int_2[1]))
    ext_line_coords = (int(ext_1[0]), int(ext_1[1]), int(ext_2[0]), int(ext_2[1]))

    cap = cv2.VideoCapture("test-video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    people = {}
    first_json_time = timestamps[0]

    if mode == 'play':
        while True:
            try:
                timestamp_delta = float(input("Введите дельту времени (в секундах, можно дробное значение):\n"))
                if timestamp_delta >= 0:
                    break
                else:
                    print("Дельту времени нельзя вводить отрицательной. Попробуйте снова.")
            except ValueError:
                print("Пожалуйста, введите число (можно с десятичной точкой).")

        print(f"Общее количество кадров: {total_frames}")
        start_frame_pos = max(0, int(timestamp_delta * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_pos)
        current_frame_pos = start_frame_pos
        out_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        def find_closest_frame(target_time):
            pos = bisect_left(timestamps, target_time)
            if pos == 0:
                return timestamps[0]
            if pos == len(timestamps):
                return timestamps[-1]
            before = timestamps[pos - 1]
            after = timestamps[pos]
            return after if after - target_time < target_time - before else before

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate video time starting from 0, adjusted for the starting frame
            current_video_time = current_frame_pos / fps
            current_frame_pos += 1
            current_datetime = first_json_time + (current_video_time - (start_frame_pos / fps))

            # Find the closest JSON frame timestamp
            closest_time = find_closest_frame(current_datetime)
            fr = frames_dict[closest_time]
            current_frame_people = set()

            if "detected" in fr and "person" in fr["detected"]:
                enter_count, exit_count = process_frame_data(fr, people, int_line_coords, ext_line_coords, current_datetime, enter_count, exit_count, current_frame_people)

                for person_data in fr["detected"]["person"]:
                    x1, y1 = int(person_data[0]), int(person_data[1])
                    x2, y2 = int(person_data[2]), int(person_data[3])
                    track_id = None
                    if len(person_data) > 5 and isinstance(person_data[5], dict):
                        for key in person_data[5].keys():
                            if 'track_id' in person_data[5][key]:
                                track_id = person_data[5][key]['track_id']
                                break
                    if track_id is not None:
                        person = people[track_id]
                        color = (0, 255, 0) if not person.entered and not person.exited else (0, 255, 255) if person.entered and not person.exited else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        status_text = "ENTERED" if person.entered and not person.exited else "EXITED" if person.exited else ""
                        cv2.putText(frame, f"ID: {track_id} {status_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.line(frame, (int_line_coords[0], int_line_coords[1]), (int_line_coords[2], int_line_coords[3]), (0, 255, 255), 2)
            cv2.putText(frame, "Entry", (int_line_coords[0], int_line_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.line(frame, (ext_line_coords[0], ext_line_coords[1]), (ext_line_coords[2], ext_line_coords[3]), (255, 0, 0), 2)
            cv2.putText(frame, "Exit", (ext_line_coords[0], ext_line_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Entered: {enter_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exited: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out_video.write(frame)
            cv2.imshow('People Counter with Track IDs', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        print("Видео с рамками сохранено в файл 'output_video.mp4'")
    else:  # count-only mode
        start_time = first_json_time
        current_frame_pos = 0

        while current_frame_pos < total_frames:
            current_video_time = start_time + (current_frame_pos / fps)
            current_frame_pos += 1
            current_datetime = current_video_time

            if current_video_time < first_json_time:
                closest_time = first_json_time
            else:
                closest_time = min(timestamps, key=lambda x: abs(x - current_video_time))

            fr = frames_dict[closest_time]
            current_frame_people = set()

            if "detected" in fr and "person" in fr["detected"]:
                enter_count, exit_count = process_frame_data(fr, people, int_line_coords, ext_line_coords, current_datetime, enter_count, exit_count, current_frame_people)

        cap.release()

    for track_id, person in people.items():
        if person.entered or person.exited:
            people_data.append({
                'Person ID': track_id,
                'Entry Time': person.entry_time if person.entry_time else "N/A",
                'Exit Time': person.exit_time if person.exit_time else "N/A"
            })

    stats_data = {
        'Total Entered': [enter_count],
        'Total Exited': [exit_count]
    }

    with pd.ExcelWriter('people_counter_report.xlsx') as writer:
        pd.DataFrame(people_data).to_excel(writer, sheet_name='People Data', index=False)
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)

    print("Отчет успешно сохранен в файл 'people_counter_report.xlsx'")
    print(f"Итоговое количество вошедших: {enter_count}")
    print(f"Итоговое количество вышедших: {exit_count}")


if __name__ == "__main__":
    main()