-- ============================================================
-- PostgreSQL version of: data/internal_database.json
-- File purpose: shared dataset for ALL interns (do not modify)
-- ============================================================

BEGIN;

-- 0) (Optional) Create schema for isolation
CREATE SCHEMA IF NOT EXISTS intern_task;
SET search_path TO intern_task;

-- 1) Metadata
CREATE TABLE IF NOT EXISTS dataset_metadata (
  id            SMALLINT PRIMARY KEY DEFAULT 1,
  version       TEXT NOT NULL,
  last_updated  TIMESTAMPTZ NOT NULL,
  description   TEXT NOT NULL,
  CONSTRAINT dataset_metadata_singleton CHECK (id = 1)
);

-- 2) Policies (normalized)
CREATE TABLE IF NOT EXISTS policies (
  policy_id    TEXT PRIMARY KEY,
  title        TEXT NOT NULL UNIQUE,
  category     TEXT NOT NULL,
  description  TEXT NOT NULL,
  role_scope   TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_rules (
  policy_id   TEXT NOT NULL REFERENCES policies(policy_id) ON DELETE CASCADE,
  rule_order  INTEGER NOT NULL CHECK (rule_order > 0),
  rule_text   TEXT NOT NULL,
  PRIMARY KEY (policy_id, rule_order)
);

-- 3) SLA Lookup
CREATE TABLE IF NOT EXISTS sla_lookup (
  service_name          TEXT PRIMARY KEY,
  tier                  TEXT NOT NULL,
  response_time         TEXT NOT NULL,
  resolution_time       TEXT NOT NULL,
  availability          TEXT NOT NULL,
  support_channels      TEXT[] NOT NULL,
  escalation_available  BOOLEAN NOT NULL
);

-- 4) Accounts
CREATE TABLE IF NOT EXISTS accounts (
  user_id      TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  role         TEXT NOT NULL,
  status       TEXT NOT NULL,
  service_plan TEXT NOT NULL REFERENCES sla_lookup(service_name),
  last_login   TIMESTAMPTZ NOT NULL
);

-- 5) System Status (singleton row)
CREATE TABLE IF NOT EXISTS system_status (
  id                    SMALLINT PRIMARY KEY DEFAULT 1,
  current_load_percentage INTEGER NOT NULL CHECK (current_load_percentage BETWEEN 0 AND 100),
  active_incidents        INTEGER NOT NULL CHECK (active_incidents >= 0),
  system_health           TEXT NOT NULL,
  maintenance_mode        BOOLEAN NOT NULL,
  last_updated            TIMESTAMPTZ NOT NULL,
  CONSTRAINT system_status_singleton CHECK (id = 1)
);

-- ============================================================
-- Seed data
-- ============================================================

-- Metadata
INSERT INTO dataset_metadata (id, version, last_updated, description)
VALUES (
  1,
  '1.0',
  '2026-02-18T00:00:00Z'::timestamptz,
  'Official shared internal database for Agentic AI Tools Implementation Task. No modification allowed.'
)
ON CONFLICT (id) DO UPDATE
SET version = EXCLUDED.version,
    last_updated = EXCLUDED.last_updated,
    description = EXCLUDED.description;

-- Policies
INSERT INTO policies (policy_id, title, category, description, role_scope) VALUES
  ('POL-001', 'Access Control Policy', 'Security',
   'Defines how system access is granted, reviewed, and revoked.',
   ARRAY['Employee','Manager','Admin']::text[]),
  ('POL-002', 'Data Deletion Policy', 'Compliance',
   'Defines rules and restrictions for deleting system data.',
   ARRAY['Admin']::text[]),
  ('POL-003', 'Incident Escalation Policy', 'Operations',
   'Defines incident severity levels and escalation timelines.',
   ARRAY['Support','Manager']::text[])
ON CONFLICT (policy_id) DO UPDATE
SET title = EXCLUDED.title,
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    role_scope = EXCLUDED.role_scope;

-- Policy rules (ordered)
-- POL-001
INSERT INTO policy_rules (policy_id, rule_order, rule_text) VALUES
  ('POL-001', 1, 'All access requests must be approved by a department manager.'),
  ('POL-001', 2, 'Production access requires security team approval.'),
  ('POL-001', 3, 'Temporary access expires automatically after 30 days.'),
  ('POL-001', 4, 'Access reviews must be conducted quarterly.')
ON CONFLICT (policy_id, rule_order) DO UPDATE
SET rule_text = EXCLUDED.rule_text;

-- POL-002
INSERT INTO policy_rules (policy_id, rule_order, rule_text) VALUES
  ('POL-002', 1, 'Bulk deletion requires two-level approval.'),
  ('POL-002', 2, 'Customer financial records cannot be deleted manually.'),
  ('POL-002', 3, 'Deletion requests must be logged and audited.'),
  ('POL-002', 4, 'Data retention minimum period is 5 years.')
ON CONFLICT (policy_id, rule_order) DO UPDATE
SET rule_text = EXCLUDED.rule_text;

-- POL-003
INSERT INTO policy_rules (policy_id, rule_order, rule_text) VALUES
  ('POL-003', 1, 'Critical incidents must be escalated within 15 minutes.'),
  ('POL-003', 2, 'Major incidents must be escalated within 1 hour.'),
  ('POL-003', 3, 'Minor incidents must be reviewed within 24 hours.'),
  ('POL-003', 4, 'All escalations must notify the on-call manager.')
ON CONFLICT (policy_id, rule_order) DO UPDATE
SET rule_text = EXCLUDED.rule_text;

-- SLA lookup
INSERT INTO sla_lookup (
  service_name, tier, response_time, resolution_time, availability, support_channels, escalation_available
) VALUES
  ('Basic Support', 'Basic', '24 hours', '3 business days', 'Business hours (09:00-18:00)', ARRAY['Email']::text[], false),
  ('Premium Support', 'Premium', '1 hour', '8 hours', '24/7', ARRAY['Email','Phone','Chat']::text[], true),
  ('Enterprise Support', 'Enterprise', '15 minutes', '4 hours', '24/7 with dedicated manager', ARRAY['Dedicated Hotline','Priority Email']::text[], true)
ON CONFLICT (service_name) DO UPDATE
SET tier = EXCLUDED.tier,
    response_time = EXCLUDED.response_time,
    resolution_time = EXCLUDED.resolution_time,
    availability = EXCLUDED.availability,
    support_channels = EXCLUDED.support_channels,
    escalation_available = EXCLUDED.escalation_available;

-- Accounts
INSERT INTO accounts (user_id, name, role, status, service_plan, last_login) VALUES
  ('1001', 'Alice Tan', 'Employee', 'Active', 'Basic Support', '2026-02-17T10:15:00Z'::timestamptz),
  ('1002', 'Brian Lim', 'Manager', 'Active', 'Premium Support', '2026-02-17T08:22:00Z'::timestamptz),
  ('1003', 'Clara Wijaya', 'Admin', 'Suspended', 'Enterprise Support', '2026-02-10T19:03:00Z'::timestamptz)
ON CONFLICT (user_id) DO UPDATE
SET name = EXCLUDED.name,
    role = EXCLUDED.role,
    status = EXCLUDED.status,
    service_plan = EXCLUDED.service_plan,
    last_login = EXCLUDED.last_login;

-- System status (singleton)
INSERT INTO system_status (
  id, current_load_percentage, active_incidents, system_health, maintenance_mode, last_updated
) VALUES (
  1, 72, 1, 'Operational', false, '2026-02-18T07:45:00Z'::timestamptz
)
ON CONFLICT (id) DO UPDATE
SET current_load_percentage = EXCLUDED.current_load_percentage,
    active_incidents = EXCLUDED.active_incidents,
    system_health = EXCLUDED.system_health,
    maintenance_mode = EXCLUDED.maintenance_mode,
    last_updated = EXCLUDED.last_updated;

COMMIT;

-- ============================================================
-- Quick sanity checks (optional)
-- ============================================================
-- SELECT * FROM intern_task.dataset_metadata;
-- SELECT * FROM intern_task.policies;
-- SELECT * FROM intern_task.policy_rules ORDER BY policy_id, rule_order;
-- SELECT * FROM intern_task.sla_lookup;
-- SELECT * FROM intern_task.accounts;
-- SELECT * FROM intern_task.system_status;
