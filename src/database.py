"""
Database module for email categorization system.
Handles SQLite database operations for storing emails and categories.
"""

import sqlite3
import pandas as pd
from typing import List, Tuple, Optional
import json
from datetime import datetime

class EmailDatabase:
    def __init__(self, db_path: str = "emails.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create emails table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_training BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                color TEXT DEFAULT '#007bff',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_sample_data(self):
        """Add comprehensive sample email data for training and testing."""
        sample_emails = [
            # Work emails
            ("Project Update Required", "Hi team, please provide updates on the Q4 project deliverables by EOD Friday.", "manager@company.com", "team@company.com", "Work"),
            ("Meeting Rescheduled", "The board meeting has been moved to next Tuesday at 2 PM in conference room A.", "admin@company.com", "all@company.com", "Work"),
            ("Budget Approval Needed", "Please review and approve the marketing budget for next quarter. Attached is the detailed breakdown.", "finance@company.com", "director@company.com", "Work"),
            ("Performance Review", "Your annual performance review is scheduled for next week. Please prepare your self-assessment.", "hr@company.com", "employee@company.com", "Work"),
            ("Client Presentation", "The client presentation went well. They're interested in expanding the project scope.", "sales@company.com", "team@company.com", "Work"),
            
            # Personal emails
            ("Birthday Party Invitation", "You're invited to Sarah's surprise birthday party this Saturday at 7 PM!", "friend@email.com", "you@email.com", "Personal"),
            ("Family Reunion", "Don't forget about the family reunion next month. Mom is making her famous apple pie!", "cousin@email.com", "you@email.com", "Personal"),
            ("Weekend Plans", "Want to go hiking this weekend? The weather looks perfect for a mountain trail.", "buddy@email.com", "you@email.com", "Personal"),
            ("Recipe Share", "Here's that chocolate cake recipe you asked for. It's been in our family for generations!", "aunt@email.com", "you@email.com", "Personal"),
            ("Book Club", "This month we're reading 'The Seven Husbands of Evelyn Hugo'. Meeting is next Thursday.", "bookclub@email.com", "members@email.com", "Personal"),
            
            # Promotions
            ("Flash Sale - 70% Off!", "Limited time offer! Get 70% off all electronics. Sale ends tonight at midnight!", "deals@store.com", "customer@email.com", "Promotions"),
            ("Exclusive Member Discount", "As a VIP member, enjoy 30% off your next purchase plus free shipping!", "vip@retailer.com", "member@email.com", "Promotions"),
            ("Black Friday Preview", "Get early access to our Black Friday deals! Shop now before everyone else.", "marketing@shop.com", "subscriber@email.com", "Promotions"),
            ("Loyalty Points Expiring", "Your 5000 loyalty points expire soon! Use them now to get amazing rewards.", "rewards@store.com", "customer@email.com", "Promotions"),
            ("New Product Launch", "Introducing our revolutionary new smartphone with AI-powered camera. Pre-order now!", "launch@tech.com", "customer@email.com", "Promotions"),
            
            # Spam
            ("Congratulations! You've Won!", "You've won $1,000,000! Click here to claim your prize now! Limited time offer!", "winner@fake.com", "victim@email.com", "Spam"),
            ("Urgent: Account Suspended", "Your account will be suspended unless you verify your information immediately!", "security@phishing.com", "target@email.com", "Spam"),
            ("Make Money Fast", "Earn $5000 per week working from home! No experience required! Start today!", "money@scam.com", "user@email.com", "Spam"),
            ("Nigerian Prince", "I am a prince who needs your help transferring millions of dollars. You will be rewarded!", "prince@nigeria.fake", "helper@email.com", "Spam"),
            ("Fake Pharmacy", "Get prescription medications without a prescription! Cheap prices, fast delivery!", "pills@illegal.com", "customer@email.com", "Spam"),
            
            # Newsletter/Updates
            ("Weekly Newsletter", "Here's your weekly roundup of tech news, product updates, and industry insights.", "newsletter@techblog.com", "subscriber@email.com", "Newsletter"),
            ("Software Update Available", "A new version of your favorite app is available with bug fixes and new features.", "updates@software.com", "user@email.com", "Newsletter"),
            ("Monthly Report", "Your monthly analytics report is ready. See how your website performed this month.", "analytics@service.com", "webmaster@email.com", "Newsletter"),
            ("Community Digest", "Top discussions from this week in our developer community forum.", "community@devforum.com", "member@email.com", "Newsletter"),
            ("Industry Trends", "Stay ahead with the latest trends in artificial intelligence and machine learning.", "insights@airesearch.com", "professional@email.com", "Newsletter"),
        ]
        
        categories = [
            ("Work", "Work-related emails including meetings, projects, and business communications", "#dc3545"),
            ("Personal", "Personal emails from friends, family, and social activities", "#28a745"),
            ("Promotions", "Marketing emails, sales, discounts, and promotional offers", "#ffc107"),
            ("Spam", "Unwanted emails, phishing attempts, and suspicious content", "#6c757d"),
            ("Newsletter", "Newsletters, updates, and informational content", "#17a2b8"),
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert categories
        cursor.executemany(
            "INSERT OR IGNORE INTO categories (name, description, color) VALUES (?, ?, ?)",
            categories
        )
        
        # Insert sample emails
        for subject, body, sender, recipient, category in sample_emails:
            cursor.execute(
                "INSERT INTO emails (subject, body, sender, recipient, category) VALUES (?, ?, ?, ?, ?)",
                (subject, body, sender, recipient, category)
            )
        
        conn.commit()
        conn.close()
        
    def get_all_emails(self) -> pd.DataFrame:
        """Retrieve all emails as a pandas DataFrame."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM emails", conn)
        conn.close()
        return df
        
    def get_training_emails(self) -> pd.DataFrame:
        """Retrieve only training emails."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM emails WHERE is_training = 1", conn)
        conn.close()
        return df
        
    def add_email(self, subject: str, body: str, sender: str, recipient: str, 
                  category: str = None, confidence: float = 0.0, is_training: bool = False):
        """Add a new email to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO emails (subject, body, sender, recipient, category, confidence, is_training) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (subject, body, sender, recipient, category, confidence, is_training)
        )
        
        conn.commit()
        conn.close()
        
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM categories")
        categories = [row[0] for row in cursor.fetchall()]
        conn.close()
        return categories
        
    def get_category_stats(self) -> pd.DataFrame:
        """Get statistics for each category."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT category, COUNT(*) as count, 
                   AVG(confidence) as avg_confidence
            FROM emails 
            GROUP BY category
        """, conn)
        conn.close()
        return df
