"""
ITS Playground - Synthetic Data Generator
Generates realistic interrelated datasets for traffic analytics practice
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
import json

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class ITSDataGenerator:
    def __init__(self):
        self.vehicle_classes = {
            2: {"name": "Car", "weight": 0.65, "speed_range": (60, 120), "tag_rate": 0.85},
            3: {"name": "Van", "weight": 0.15, "speed_range": (50, 100), "tag_rate": 0.90},
            4: {"name": "Truck", "weight": 0.12, "speed_range": (40, 80), "tag_rate": 0.95},
            5: {"name": "Bus", "weight": 0.05, "speed_range": (45, 75), "tag_rate": 0.92},
            6: {"name": "Heavy Truck", "weight": 0.03, "speed_range": (35, 65), "tag_rate": 0.98}
        }
        
        self.toll_rates = {2: 3.50, 3: 5.25, 4: 8.75, 5: 7.00, 6: 12.25}
        self.payment_methods = ["Tag", "Credit_Card", "Cash", "Mobile_App"]
        self.lanes = list(range(1, 9))  # 8 lanes
        self.directions = ["North", "South"]
        
    def generate_passage_logs(self, num_records=10000, start_date="2024-01-01"):
        """Generate vehicle passage logs with realistic patterns"""
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        records = []
        
        for i in range(num_records):
            # Generate realistic timestamp with traffic patterns
            days_offset = random.randint(0, 89)  # 3 months of data
            hour_weights = [0.3, 0.2, 0.15, 0.1, 0.2, 0.4, 0.8, 1.0, 0.9, 0.7, 0.6, 0.7,
                           0.8, 0.7, 0.6, 0.7, 0.9, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3]
            hour = np.random.choice(24, p=np.array(hour_weights)/sum(hour_weights))
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            
            timestamp = start_dt + timedelta(days=days_offset, hours=hour, minutes=minute, seconds=second)
            
            # Select vehicle class based on weights
            classes, weights = zip(*[(k, v["weight"]) for k, v in self.vehicle_classes.items()])
            axle_count = np.random.choice(classes, p=weights)
            vehicle_class = self.vehicle_classes[axle_count]["name"]
            
            # Generate speed based on class and add some variance
            speed_min, speed_max = self.vehicle_classes[axle_count]["speed_range"]
            speed = round(random.uniform(speed_min, speed_max), 1)
            
            # Tag detection based on class rates
            has_tag = random.random() < self.vehicle_classes[axle_count]["tag_rate"]
            tag_id = f"TAG_{random.randint(100000, 999999)}" if has_tag else None
            
            # Lane assignment with some bias towards middle lanes
            lane_weights = [0.08, 0.12, 0.18, 0.22, 0.22, 0.18, 0.12, 0.08]
            lane_number = np.random.choice(self.lanes, p=lane_weights)
            
            # GPS coordinates (example highway coordinates)
            base_lat, base_lon = 40.7589, -73.9851  # Example coordinates
            gps_lat = round(base_lat + random.uniform(-0.001, 0.001), 6)
            gps_lon = round(base_lon + random.uniform(-0.001, 0.001), 6)
            
            # Generate unique IDs
            passage_id = f"PASS_{str(uuid.uuid4())[:8].upper()}"
            vehicle_id = f"VEH_{random.randint(100000, 999999)}"
            
            record = {
                "passage_id": passage_id,
                "vehicle_id": vehicle_id,
                "timestamp": timestamp,
                "axle_count": axle_count,
                "vehicle_class": vehicle_class,
                "speed_kmh": speed,
                "lane_number": lane_number,
                "direction": random.choice(self.directions),
                "tag_id": tag_id,
                "tag_detected": has_tag,
                "gps_latitude": gps_lat,
                "gps_longitude": gps_lon,
                "sensor_id": f"SENSOR_{lane_number:02d}",
                "image_captured": random.random() < 0.95,  # 95% image capture rate
                "trip_start": random.random() < 0.7,  # 70% are trip starts
            }
            
            records.append(record)
        
        return pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    
    def generate_transaction_data(self, passage_df):
        """Generate toll transaction data based on passage logs"""
        
        transactions = []
        
        for _, passage in passage_df.iterrows():
            # Only create transaction if tag detected or manual payment
            if passage['tag_detected'] or random.random() < 0.15:  # 15% manual payments for non-tag
                
                # Determine payment method
                if passage['tag_detected']:
                    payment_method = "Tag"
                else:
                    payment_method = np.random.choice(
                        ["Credit_Card", "Cash", "Mobile_App"], 
                        p=[0.5, 0.3, 0.2]
                    )
                
                # Calculate toll amount
                base_amount = self.toll_rates[passage['axle_count']]
                # Add surge pricing during peak hours
                hour = passage['timestamp'].hour
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    surge_multiplier = 1.2
                else:
                    surge_multiplier = 1.0
                
                toll_amount = round(base_amount * surge_multiplier, 2)
                
                # Transaction status
                success_rate = 0.98 if payment_method == "Tag" else 0.95
                transaction_status = "Success" if random.random() < success_rate else "Failed"
                
                # Processing time varies by payment method
                processing_times = {"Tag": 0.5, "Credit_Card": 3.2, "Cash": 8.5, "Mobile_App": 2.1}
                processing_time = round(
                    processing_times[payment_method] + random.uniform(-0.5, 0.5), 2
                )
                
                transaction = {
                    "transaction_id": f"TXN_{str(uuid.uuid4())[:8].upper()}",
                    "passage_id": passage['passage_id'],
                    "vehicle_id": passage['vehicle_id'],
                    "timestamp": passage['timestamp'] + timedelta(seconds=processing_time),
                    "toll_amount": toll_amount,
                    "payment_method": payment_method,
                    "transaction_status": transaction_status,
                    "processing_time_sec": processing_time,
                    "lane_number": passage['lane_number'],
                    "vehicle_class": passage['vehicle_class'],
                    "tag_id": passage['tag_id']
                }
                
                transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def generate_anpr_data(self, passage_df):
        """Generate ANPR/IVDC (Automatic Number Plate Recognition) data"""
        
        anpr_records = []
        
        for _, passage in passage_df.iterrows():
            if passage['image_captured']:
                # Generate synthetic license plate
                states = ["NY", "CA", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
                state = random.choice(states)
                plate_number = f"{state}{random.randint(100, 999)}{random.choice(['ABC', 'XYZ', 'DEF', 'GHI'])}"
                
                # Image quality factors
                weather_conditions = ["Clear", "Rain", "Snow", "Fog"]
                weather = np.random.choice(weather_conditions, p=[0.7, 0.15, 0.1, 0.05])
                
                # OCR confidence based on conditions
                base_confidence = 0.95
                if weather == "Rain":
                    base_confidence *= 0.9
                elif weather == "Snow":
                    base_confidence *= 0.85
                elif weather == "Fog":
                    base_confidence *= 0.8
                
                # Night time reduces confidence
                if passage['timestamp'].hour < 6 or passage['timestamp'].hour > 20:
                    base_confidence *= 0.9
                
                ocr_confidence = round(base_confidence + random.uniform(-0.1, 0.05), 3)
                
                anpr_record = {
                    "anpr_id": f"ANPR_{str(uuid.uuid4())[:8].upper()}",
                    "passage_id": passage['passage_id'],
                    "vehicle_id": passage['vehicle_id'],
                    "timestamp": passage['timestamp'],
                    "license_plate": plate_number,
                    "state": state,
                    "ocr_confidence": ocr_confidence,
                    "image_quality": "High" if ocr_confidence > 0.9 else "Medium" if ocr_confidence > 0.8 else "Low",
                    "weather_condition": weather,
                    "camera_id": f"CAM_{passage['lane_number']:02d}",
                    "vehicle_make": random.choice(["Toyota", "Ford", "Honda", "Chevrolet", "BMW", "Mercedes", "Audi"]),
                    "vehicle_color": random.choice(["White", "Black", "Silver", "Blue", "Red", "Gray"]),
                    "match_confidence": round(random.uniform(0.85, 0.99), 3)
                }
                
                anpr_records.append(anpr_record)
        
        return pd.DataFrame(anpr_records)
    
    def generate_ev_charging_data(self, num_records=5000):
        """Generate EV charging station data"""
        
        charging_records = []
        start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
        
        for i in range(num_records):
            # Charging session timing
            days_offset = random.randint(0, 89)
            hour_weights = [0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.4, 0.6, 0.5, 0.4, 0.3, 0.3,
                           0.4, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
            hour = np.random.choice(24, p=np.array(hour_weights)/sum(hour_weights))
            
            start_time = start_date + timedelta(days=days_offset, hours=hour, minutes=random.randint(0, 59))
            
            # Charging duration (minutes)
            charging_duration = random.randint(15, 180)  # 15 minutes to 3 hours
            end_time = start_time + timedelta(minutes=charging_duration)
            
            # Energy consumption
            power_rating = random.choice([7.2, 11, 22, 50, 150, 350])  # kW
            energy_delivered = round(power_rating * (charging_duration / 60) * random.uniform(0.8, 1.0), 2)
            
            charging_record = {
                "session_id": f"EVCS_{str(uuid.uuid4())[:8].upper()}",
                "station_id": f"STATION_{random.randint(1, 20):03d}",
                "charger_id": f"CHG_{random.randint(1, 100):03d}",
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": charging_duration,
                "energy_delivered_kwh": energy_delivered,
                "power_rating_kw": power_rating,
                "charging_cost": round(energy_delivered * random.uniform(0.12, 0.35), 2),
                "payment_method": random.choice(["Credit_Card", "Mobile_App", "RFID_Card"]),
                "vehicle_type": random.choice(["Tesla", "BMW", "Audi", "Nissan", "Chevrolet"]),
                "user_id": f"USER_{random.randint(10000, 99999)}",
                "station_location": random.choice(["Highway_Rest", "Shopping_Mall", "Office_Complex", "Residential"]),
                "grid_impact_kw": round(power_rating * random.uniform(0.9, 1.1), 2)
            }
            
            charging_records.append(charging_record)
        
        return pd.DataFrame(charging_records).sort_values('start_time').reset_index(drop=True)
    
    def generate_road_safety_data(self, passage_df, num_incidents=200):
        """Generate road safety incident data"""
        
        incidents = []
        
        for i in range(num_incidents):
            # Select random passage as base for incident location
            base_passage = passage_df.sample(1).iloc[0]
            
            incident_types = ["Accident", "Breakdown", "Debris", "Weather", "Construction"]
            severity_levels = ["Low", "Medium", "High", "Critical"]
            
            incident_type = random.choice(incident_types)
            severity = np.random.choice(severity_levels, p=[0.4, 0.35, 0.2, 0.05])
            
            # Incident duration varies by type and severity
            duration_ranges = {
                "Low": (5, 30), "Medium": (15, 60), 
                "High": (30, 120), "Critical": (60, 300)
            }
            duration = random.randint(*duration_ranges[severity])
            
            incident = {
                "incident_id": f"INC_{str(uuid.uuid4())[:8].upper()}",
                "timestamp": base_passage['timestamp'],
                "incident_type": incident_type,
                "severity": severity,
                "lane_number": base_passage['lane_number'],
                "direction": base_passage['direction'],
                "gps_latitude": base_passage['gps_latitude'],
                "gps_longitude": base_passage['gps_longitude'],
                "duration_minutes": duration,
                "vehicles_involved": random.randint(1, 4),
                "injuries": random.randint(0, 3) if incident_type == "Accident" else 0,
                "emergency_response": random.random() < 0.7,
                "traffic_impact": random.choice(["None", "Light", "Moderate", "Heavy"]),
                "weather_condition": random.choice(["Clear", "Rain", "Snow", "Fog"])
            }
            
            incidents.append(incident)
        
        return pd.DataFrame(incidents).sort_values('timestamp').reset_index(drop=True)

def generate_all_datasets():
    """Generate all interconnected datasets"""
    
    print("🚦 ITS Playground Data Generator")
    print("=" * 40)
    
    generator = ITSDataGenerator()
    
    # Generate core datasets
    print("📊 Generating passage logs...")
    passage_df = generator.generate_passage_logs(num_records=10000)
    
    print("💳 Generating transaction data...")
    transaction_df = generator.generate_transaction_data(passage_df)
    
    print("📷 Generating ANPR data...")
    anpr_df = generator.generate_anpr_data(passage_df)
    
    print("⚡ Generating EV charging data...")
    ev_df = generator.generate_ev_charging_data(num_records=5000)
    
    print("🚨 Generating road safety data...")
    safety_df = generator.generate_road_safety_data(passage_df, num_incidents=200)
    
    # Save all datasets
    datasets = {
        "passage_logs": passage_df,
        "toll_transactions": transaction_df,
        "anpr_data": anpr_df,
        "ev_charging": ev_df,
        "road_safety_incidents": safety_df
    }
    
    print("\n📁 Dataset Summary:")
    for name, df in datasets.items():
        print(f"  • {name}: {len(df):,} records")
    
    # Save as CSV files
    for name, df in datasets.items():
        df.to_csv(f"{name}.csv", index=False)
        print(f"✅ Saved: {name}.csv")
    
    # Generate data quality report
    print("\n📋 Data Quality Report:")
    print(f"  • Total Passages: {len(passage_df):,}")
    print(f"  • Successful Transactions: {len(transaction_df[transaction_df['transaction_status'] == 'Success']):,}")
    print(f"  • Tag Detection Rate: {passage_df['tag_detected'].mean():.1%}")
    print(f"  • ANPR Success Rate: {anpr_df['ocr_confidence'].mean():.1%}")
    print(f"  • Average Charging Duration: {ev_df['duration_minutes'].mean():.1f} minutes")
    
    return datasets

# Generate sample data relationships
def create_sample_analysis():
    """Create sample analysis queries to demonstrate data relationships"""
    
    analysis_queries = {
        "traffic_kpis": """
        # Traffic Flow KPIs
        SELECT 
            DATE(timestamp) as date,
            HOUR(timestamp) as hour,
            COUNT(*) as vehicle_count,
            AVG(speed_kmh) as avg_speed,
            COUNT(DISTINCT vehicle_class) as class_variety
        FROM passage_logs 
        GROUP BY DATE(timestamp), HOUR(timestamp)
        ORDER BY date, hour
        """,
        
        "revenue_analysis": """
        # Revenue Analysis by Payment Method
        SELECT 
            payment_method,
            COUNT(*) as transaction_count,
            SUM(toll_amount) as total_revenue,
            AVG(toll_amount) as avg_toll,
            AVG(processing_time_sec) as avg_processing_time
        FROM toll_transactions 
        WHERE transaction_status = 'Success'
        GROUP BY payment_method
        """,
        
        "anomaly_detection": """
        # Detect Missing Tag Transactions
        SELECT 
            p.passage_id,
            p.vehicle_id,
            p.timestamp,
            p.vehicle_class,
            p.tag_detected,
            t.transaction_id
        FROM passage_logs p
        LEFT JOIN toll_transactions t ON p.passage_id = t.passage_id
        WHERE p.tag_detected = FALSE AND t.transaction_id IS NULL
        """,
        
        "ev_grid_impact": """
        # EV Charging Grid Impact Analysis
        SELECT 
            HOUR(start_time) as hour,
            station_location,
            COUNT(*) as charging_sessions,
            SUM(energy_delivered_kwh) as total_energy,
            AVG(grid_impact_kw) as avg_grid_impact
        FROM ev_charging
        GROUP BY HOUR(start_time), station_location
        ORDER BY hour, station_location
        """
    }
    
    return analysis_queries

if __name__ == "__main__":
    # Generate all datasets
    datasets = generate_all_datasets()
    
    # Create analysis examples
    queries = create_sample_analysis()
    
    # Save analysis queries as JSON
    with open("sample_analysis_queries.json", "w") as f:
        json.dump(queries, f, indent=2)
    
    print(f"\n🎯 Sample analysis queries saved to: sample_analysis_queries.json")
    print(f"🚀 Ready to start your ITS analytics journey!")
