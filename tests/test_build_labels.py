"""
Unit tests for build_labels.py

Tests the label building logic with in-memory data to ensure deterministic behavior.
"""

import unittest
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

# Add execution directory to path to import build_labels
sys.path.insert(0, str(Path(__file__).parent.parent / 'execution'))
from build_labels import build_labels, format_iso_timestamp


class TestBuildLabels(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with timezone-aware UTC datetimes."""
        # Base engagement timestamp
        self.base_engagement_time = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        # Create engagement data (contact_id -> first_engagement_timestamp)
        self.engagement_data = {
            'contact_001': self.base_engagement_time,
            'contact_002': self.base_engagement_time + timedelta(hours=1),
            'contact_003': self.base_engagement_time + timedelta(hours=2),
            'contact_004': self.base_engagement_time + timedelta(hours=3)
        }
    
    def test_booking_inside_window_returns_label_1(self):
        """Test that booking within 7-day window generates label 1."""
        # Booking 3 days after engagement (inside 7-day window)
        outcome_data = {
            'contact_001': self.base_engagement_time + timedelta(days=3)
        }
        
        labels = build_labels(self.engagement_data, outcome_data, label_window_days=7)
        
        # Find contact_001 label
        contact_001_label = next(label for label in labels if label['ghl_contact_id'] == 'contact_001')
        
        self.assertEqual(contact_001_label['booked_call_within_7d'], 1)
        self.assertEqual(contact_001_label['ghl_contact_id'], 'contact_001')
        self.assertNotEqual(contact_001_label['booking_created_at'], "")  # Should have booking timestamp
    
    def test_booking_after_window_returns_label_0(self):
        """Test that booking after 7-day window generates label 0."""
        # Booking 10 days after engagement (outside 7-day window)
        outcome_data = {
            'contact_002': self.base_engagement_time + timedelta(days=10)
        }
        
        labels = build_labels(self.engagement_data, outcome_data, label_window_days=7)
        
        # Find contact_002 label
        contact_002_label = next(label for label in labels if label['ghl_contact_id'] == 'contact_002')
        
        self.assertEqual(contact_002_label['booked_call_within_7d'], 0)
        self.assertEqual(contact_002_label['ghl_contact_id'], 'contact_002')
        self.assertNotEqual(contact_002_label['booking_created_at'], "")  # Should have booking timestamp
    
    def test_no_booking_returns_label_0(self):
        """Test that no booking generates label 0 with empty booking timestamp."""
        # Empty outcome data (no bookings)
        outcome_data = {}
        
        labels = build_labels(self.engagement_data, outcome_data, label_window_days=7)
        
        # Find contact_003 label (should have no booking)
        contact_003_label = next(label for label in labels if label['ghl_contact_id'] == 'contact_003')
        
        self.assertEqual(contact_003_label['booked_call_within_7d'], 0)
        self.assertEqual(contact_003_label['ghl_contact_id'], 'contact_003')
        self.assertEqual(contact_003_label['booking_created_at'], "")  # Should be empty
    
    def test_output_sorted_by_contact_id(self):
        """Test that output is deterministically sorted by ghl_contact_id."""
        # Mixed outcome data
        outcome_data = {
            'contact_004': self.base_engagement_time + timedelta(days=2),
            'contact_002': self.base_engagement_time + timedelta(days=1)
        }
        
        labels = build_labels(self.engagement_data, outcome_data, label_window_days=7)
        
        # Extract contact IDs in order
        contact_ids = [label['ghl_contact_id'] for label in labels]
        
        # Should be sorted alphabetically
        expected_order = ['contact_001', 'contact_002', 'contact_003', 'contact_004']
        self.assertEqual(contact_ids, expected_order)
        
        # Verify we have all contacts from engagement data
        self.assertEqual(len(labels), len(self.engagement_data))
    
    def test_edge_case_booking_exactly_at_window_end(self):
        """Test booking exactly at window boundary (should be excluded)."""
        # Booking exactly 7 days after engagement (at window boundary)
        outcome_data = {
            'contact_001': self.base_engagement_time + timedelta(days=7)
        }
        
        labels = build_labels(self.engagement_data, outcome_data, label_window_days=7)
        
        # Find contact_001 label
        contact_001_label = next(label for label in labels if label['ghl_contact_id'] == 'contact_001')
        
        # Should be 0 because booking is at window end (exclusive)
        self.assertEqual(contact_001_label['booked_call_within_7d'], 0)
    
    def test_booking_before_engagement_returns_label_0(self):
        """Test that booking before engagement generates label 0."""
        # Booking 1 day before engagement
        outcome_data = {
            'contact_001': self.base_engagement_time - timedelta(days=1)
        }
        
        labels = build_labels(self.engagement_data, outcome_data, label_window_days=7)
        
        # Find contact_001 label
        contact_001_label = next(label for label in labels if label['ghl_contact_id'] == 'contact_001')
        
        self.assertEqual(contact_001_label['booked_call_within_7d'], 0)
        self.assertNotEqual(contact_001_label['booking_created_at'], "")  # Should have booking timestamp
    
    def test_different_window_sizes(self):
        """Test that different window sizes work correctly."""
        # Booking 2 days after engagement
        outcome_data = {
            'contact_001': self.base_engagement_time + timedelta(days=2)
        }
        
        # Test with 1-day window (should be 0)
        labels_1day = build_labels(self.engagement_data, outcome_data, label_window_days=1)
        contact_001_1day = next(label for label in labels_1day if label['ghl_contact_id'] == 'contact_001')
        self.assertEqual(contact_001_1day['booked_call_within_7d'], 0)
        
        # Test with 3-day window (should be 1)
        labels_3day = build_labels(self.engagement_data, outcome_data, label_window_days=3)
        contact_001_3day = next(label for label in labels_3day if label['ghl_contact_id'] == 'contact_001')
        self.assertEqual(contact_001_3day['booked_call_within_7d'], 1)
    
    def test_format_iso_timestamp(self):
        """Test ISO timestamp formatting helper function."""
        test_dt = datetime(2026, 2, 1, 15, 30, 45, tzinfo=timezone.utc)
        formatted = format_iso_timestamp(test_dt)
        self.assertEqual(formatted, "2026-02-01T15:30:45Z")
        
        # Test None input
        formatted_none = format_iso_timestamp(None)
        self.assertEqual(formatted_none, "")


if __name__ == '__main__':
    unittest.main()