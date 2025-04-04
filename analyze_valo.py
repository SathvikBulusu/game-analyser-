import cv2
import pytesseract
import os
import psycopg2
import re
import numpy as np
from flask import Flask, render_template
import plotly.express as px
import pandas as pd
from threading import Thread
import socketserver

print("Starting script...")

# Database setup
try:
    conn = psycopg2.connect(
        dbname="valorent_tool", user="postgres", password="1234", host="localhost", port="5432"
    )
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS player_stats")
    cursor.execute("""CREATE TABLE IF NOT EXISTS player_stats (
                    id SERIAL PRIMARY KEY, video_name VARCHAR(255), round_number INT, kills INT, deaths INT, assists INT,
                    kd_ratio FLOAT, suggestions TEXT, timestamp INT, credits INT NULL, loadout INT NULL, ping INT NULL)""")
    conn.commit()
    print("Database connected and schema updated.")
except Exception as e:
    print(f"Database error: {e}")
    exit()

def find_kda_frame(video_path, sample_interval=1):
    print(f"Attempting to process video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps
    print(f"Video FPS: {fps}, Total frames: {total_frames}, Total seconds: {total_seconds}")

    stats_list = []
    current_round = 1
    last_kda = {'kills': 0, 'deaths': 0, 'assists': 0}

    os.makedirs("ocr_debug", exist_ok=True)

    for seconds in range(0, int(total_seconds), sample_interval):
        frame_num = int(fps * seconds)
        if frame_num >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at {seconds} seconds in {video_path}")
            continue

        # Enhanced preprocessing for better OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Adjust cropping to focus on scoreboard (experiment with different regions)
        gray = gray[h//4:3*h//4, w//4:3*w//4]
        # Increase contrast and sharpness
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Apply adaptive thresholding with larger block size for better text isolation
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        # Reduce noise with a slight blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Save preprocessed frame for debugging
        cv2.imwrite(f"ocr_debug/preprocessed_{os.path.basename(video_path)}_{seconds}.jpg", gray)

        # Perform OCR with multiple PSM modes for better detection
        text = pytesseract.image_to_string(gray, config='--psm 6')
        if not text.strip():
            text = pytesseract.image_to_string(gray, config='--psm 11')
        with open(f"ocr_debug/ocr_output_{os.path.basename(video_path)}_{seconds}.txt", "w") as f:
            f.write(text)
        print(f"OCR output at {seconds} seconds in {video_path}: '{text}'")

        # Detect round number
        round_match = re.search(r'Round (\d+)', text, re.IGNORECASE)
        if round_match:
            new_round = int(round_match.group(1))
            if new_round != current_round:
                print(f"Detected new round {new_round} at {seconds} seconds in {video_path}")
                current_round = new_round
                last_kda = {'kills': 0, 'deaths': 0, 'assists': 0}

        stats = None
        for line in text.splitlines():
            line_lower = line.lower().strip()
            # Broaden the search for "Me" to handle OCR errors
            me_variants = ["me", "m3", "m e", "rn3", "rne"]
            if any(variant in line_lower for variant in me_variants):
                print(f"Found potential 'Me' in line: '{line}' in {video_path}")
                # Split on spaces or slashes, and clean up
                parts = re.split(r'[\s/]+', line)
                parts_lower = [p.lower().strip() for p in parts if p.strip()]
                print(f"Parts after splitting: {parts}")
                print(f"Parts_lower: {parts_lower}")

                try:
                    me_idx = next((i for i, p in enumerate(parts_lower) if p in me_variants), None)
                    if me_idx is None:
                        print(f"Warning: No 'me' variant found in parts_lower: {parts_lower}, attempting numeric fallback in {video_path}")
                        numbers = [int(p) for p in parts if p.isdigit()][:3]
                        if len(numbers) >= 3:
                            kills, deaths, assists = numbers
                            stats = {'kills': kills, 'deaths': deaths, 'assists': assists, 'round_number': current_round}
                    else:
                        idx = me_idx
                        numbers = [int(p) for p in parts[idx+1:] if p.isdigit()][:3]
                        if len(numbers) >= 3:
                            kills, deaths, assists = numbers
                            if kills > 50 or deaths > 50 or assists > 50:
                                print(f"Invalid K/D/A values at {seconds} seconds in {video_path}: {kills}/{deaths}/{assists}, skipping.")
                                continue
                            last_kda['kills'] = max(last_kda['kills'], kills)
                            last_kda['deaths'] = max(last_kda['deaths'], deaths)
                            last_kda['assists'] = max(last_kda['assists'], assists)
                            stats = {
                                'kills': last_kda['kills'],
                                'deaths': last_kda['deaths'],
                                'assists': last_kda['assists'],
                                'round_number': current_round
                            }
                            break
                except (ValueError, IndexError) as e:
                    print(f"Error parsing K/D/A from line: '{line}' in {video_path}, error: {e}")
                    continue

        if stats:
            credits_match = re.search(r'(?:creds|credits)[:\s]*(\d+)', text, re.IGNORECASE)
            loadout_match = re.search(r'(?:loadout|load)[:\s]*(\d+)', text, re.IGNORECASE)
            ping_match = re.search(r'(?:ping)[:\s]*(\d+)', text, re.IGNORECASE)
            stats['credits'] = int(credits_match.group(1)) if credits_match and int(credits_match.group(1)) < 10000 else None
            stats['loadout'] = int(loadout_match.group(1)) if loadout_match and int(loadout_match.group(1)) < 10000 else None
            stats['ping'] = int(ping_match.group(1)) if ping_match and int(ping_match.group(1)) < 1000 else None

            if stats_list:
                last_timestamp, last_stats, _ = stats_list[-1]
                if (seconds - last_timestamp) < 5 and last_stats['kills'] == stats['kills'] and last_stats['deaths'] == stats['deaths'] and last_stats['assists'] == stats['assists']:
                    print(f"Skipping duplicate stats at {seconds} seconds in {video_path}: {stats}")
                    continue
            stats_list.append((seconds, stats, frame))
            print(f"Found stats at {seconds} seconds in {video_path}: {stats}")

    cap.release()
    print(f"Finished processing {video_path}. Stats found: {len(stats_list)} entries")
    return stats_list

def analyze_trends(video_file, conn, cursor):
    try:
        conn.rollback()
        cursor.execute("SELECT timestamp, round_number, kills, deaths, assists, credits, loadout, ping FROM player_stats WHERE video_name = %s ORDER BY timestamp", (video_file,))
        rows = cursor.fetchall()
        if len(rows) < 2:
            print(f"Skipping trend analysis for {video_file}: Not enough data points ({len(rows)}).")
            return

        insights = {}
        for i in range(len(rows)):
            t, r, k, d, a, c, l, p = rows[i]
            if r not in insights:
                insights[r] = []
            if d > 5:
                insights[r].append(f"Round {r}: Too many deaths ({d})—play more defensively.")
            if k < 2 and d > 3:
                insights[r].append(f"Round {r}: Low kills ({k}) and high deaths ({d})—focus on staying alive.")
            if p and p > 100:
                insights[r].append(f"Round {r}: High ping ({p} ms) at {t} seconds—check your network.")
            if c and c < 1000:
                insights[r].append(f"Round {r}: Low credits ({c}) at {t} seconds—consider saving more.")

        for i in range(1, len(rows)):
            t1, r1, k1, d1, a1, c1, l1, p1 = rows[i-1]
            t2, r2, k2, d2, a2, c2, l2, p2 = rows[i]
            if t1 is None or t2 is None or r1 != r2:
                continue

            time_diff = (t2 - t1) / 60
            if time_diff > 0:
                kill_rate = (k2 - k1) / time_diff
                if kill_rate < 0.5 and d2 - d1 <= 3:
                    insights[r2].append(f"Round {r2}: Low kill rate ({kill_rate:.2f} kills/min) between {t1} and {t2} seconds—try to engage more.")
                elif kill_rate > 2:
                    insights[r2].append(f"Round {r2}: High kill rate ({kill_rate:.2f} kills/min) between {t1} and {t2} seconds—great job!")
            if d2 - d1 > 3:
                insights[r2].append(f"Round {r2}: Death spike detected between {t1} and {t2} seconds: {d2 - d1} deaths—be more cautious.")

        if any(insights.values()):
            cursor.execute("SELECT suggestions FROM player_stats WHERE video_name = %s ORDER BY timestamp DESC LIMIT 1", (video_file,))
            current_suggestions = cursor.fetchone()[0] or ""
            new_suggestions = current_suggestions + "; " + "; ".join([s for sublist in insights.values() for s in sublist]) if current_suggestions else "; ".join([s for sublist in insights.values() for s in sublist])
            cursor.execute("UPDATE player_stats SET suggestions = %s WHERE video_name = %s AND timestamp = (SELECT MAX(timestamp) FROM player_stats WHERE video_name = %s)", 
                           (new_suggestions, video_file, video_file))
            conn.commit()
            print(f"Added trend insights for {video_file}: {new_suggestions}")
    except Exception as e:
        print(f"Error in analyze_trends for {video_file}: {e}")
        conn.rollback()

# Process videos
videos_dir = "videos"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for video_file in os.listdir(videos_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(videos_dir, video_file)
        print(f"\nProcessing {video_file} at path {video_path}...")

        stats_list = find_kda_frame(video_path)
        if not stats_list:
            print(f"Warning: No K/D/A screen found in {video_file}, skipping database insertion")
            continue

        for i, (timestamp, stats, frame) in enumerate(stats_list):
            if i == 0 or i == len(stats_list) - 1:
                frame_path = os.path.join(output_dir, f"{video_file}_kda_frame_{timestamp}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"K/D/A frame found at {timestamp} seconds, saved to {frame_path}")

            kd_ratio = stats["kills"] / max(stats["deaths"], 1)
            suggestions = []
            if kd_ratio < 0.8:
                suggestions.append(f"Round {stats['round_number']}: Focus on improving your K/D ratio.")
            if stats["assists"] > stats["kills"] and stats["assists"] > 5:
                suggestions.append(f"Round {stats['round_number']}: Great teamwork—keep supporting your team!")
            if stats["deaths"] > 15:
                suggestions.append(f"Round {stats['round_number']}: Too many deaths—play more defensively.")

            try:
                cursor.execute(
                    """INSERT INTO player_stats (video_name, round_number, kills, deaths, assists, kd_ratio, suggestions, timestamp, credits, loadout, ping)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (video_file, stats["round_number"], stats["kills"], stats["deaths"], stats["assists"], kd_ratio, "; ".join(suggestions), timestamp, 
                     stats.get("credits"), stats.get("loadout"), stats.get("ping"))
                )
                conn.commit()
                print(f"Stats stored for {video_file} at {timestamp} seconds: {stats}")
            except Exception as e:
                print(f"Database insert error for {video_file} at {timestamp} seconds: {e}")
                print(f"Failed stats: {stats}")
                conn.rollback()

        analyze_trends(video_file, conn, cursor)

# Flask app for UI
app = Flask(__name__)

@app.route('/')
def display_stats():
    try:
        conn_flask = psycopg2.connect(
            dbname="valorent_tool", user="postgres", password="1234", host="localhost", port="5432"
        )
        cursor_flask = conn_flask.cursor()
        cursor_flask.execute("SELECT * FROM player_stats ORDER BY video_name, timestamp")
        rows = cursor_flask.fetchall()
        columns = [desc[0] for desc in cursor_flask.description]
        print(f"Columns retrieved from database: {columns}")
        df = pd.DataFrame(rows, columns=columns)

        print(f"Initial DataFrame shape: {df.shape}")
        print(f"Initial DataFrame index: {df.index.tolist()}")
        print(f"Initial DataFrame columns: {df.columns.tolist()}")
        print(f"Initial DataFrame content:\n{df}")

        # Clean data for plotting
        df = df.dropna(subset=['timestamp'])
        df['kills'] = pd.to_numeric(df['kills'], errors='coerce').fillna(0).astype(int)
        df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce').fillna(0).astype(int)
        df['assists'] = pd.to_numeric(df['assists'], errors='coerce').fillna(0).astype(int)
        df = df[(df['kills'] <= 50) & (df['deaths'] <= 50) & (df['assists'] <= 50)]

        print(f"After cleaning, DataFrame shape: {df.shape}")
        print(f"After cleaning, DataFrame content:\n{df}")

        # Sort and reset index
        df = df.sort_values(['video_name', 'timestamp']).reset_index(drop=True)
        print(f"After sorting and resetting index, DataFrame shape: {df.shape}")
        print(f"After sorting, DataFrame content:\n{df}")

        # Interpolate to smooth the graph
        df[['kills', 'deaths', 'assists']] = df[['kills', 'deaths', 'assists']].interpolate(method='linear')
        print(f"After interpolation, DataFrame shape: {df.shape}")
        print(f"After interpolation, DataFrame content:\n{df}")

        # Create a line plot for K/D/A over time for each video
        fig = px.line(df, x="timestamp", y=["kills", "deaths", "assists"], color="video_name", 
                      title="K/D/A Trends Over Time", labels={"timestamp": "Time (seconds)", "value": "Count", "variable": "Metric"},
                      facet_col="round_number", facet_col_wrap=3)
        fig.update_yaxes(range=[0, 50])
        graph_html = fig.to_html(full_html=False)

        # Calculate summary stats
        summary = df.groupby('video_name').agg({
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
            'kd_ratio': 'mean'
        }).reset_index()
        summary.columns = ['Video Name', 'Avg Kills', 'Avg Deaths', 'Avg Assists', 'Avg K/D Ratio']
        summary = summary.round(2)
        print(f"Summary stats:\n{summary}")

        cursor_flask.close()
        conn_flask.close()

        return render_template("stats.html", tables=[df.to_html(classes='data', index=False)], 
                             summary=[summary.to_html(classes='data', index=False)], graph=graph_html)
    except Exception as e:
        return f"Error displaying stats: {e}"

# HTML template for Flask
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Valorant Stats</title>
    <style>
        table.data { border-collapse: collapse; width: 100%; }
        table.data, th, td { border: 1px solid black; padding: 5px; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Valorant Gameplay Stats</h1>
    <h2>Summary Stats</h2>
    {% for table in summary %}
        {{ table | safe }}
    {% endfor %}
    <h2>Stats Table</h2>
    {% for table in tables %}
        {{ table | safe }}
    {% endfor %}
    <h2>K/D/A Trends</h2>
    {{ graph | safe }}
</body>
</html>
"""

# Save the HTML template
os.makedirs("templates", exist_ok=True)
with open("templates/stats.html", "w") as f:
    f.write(html_template)

# Run Flask app in a separate thread
def run_flask():
    port = 5001
    for p in range(5001, 5010):
        try:
            app.run(debug=False, use_reloader=False, port=p)
            print(f"Flask app running on port {p}")
            break
        except OSError as e:
            print(f"Port {p} is in use, trying next port...")
            port = p + 1
    else:
        print("Error: Could not find an available port between 5001 and 5009.")

flask_thread = Thread(target=run_flask)
flask_thread.start()

# Clean up main database connection
cursor.close()
conn.close()
print(f"All videos processed! Access the stats at http://127.0.0.1:{port}/")