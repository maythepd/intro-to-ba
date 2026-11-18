CREATE SCHEMA IF NOT EXISTS bank_monitoring;


CREATE TABLE IF NOT EXISTS bank_monitoring.models (
	id VARCHAR(64) NOT NULL,
	created_at VARCHAR(32) NOT NULL,
	reference_sha256 VARCHAR(64) NOT NULL,
	reference JSON NOT NULL,
	details JSON NOT NULL,
	PRIMARY KEY (id)
)

;


CREATE TABLE IF NOT EXISTS bank_monitoring.schema_version (
	version SERIAL NOT NULL,
	PRIMARY KEY (version)
)

;


CREATE TABLE IF NOT EXISTS bank_monitoring.prediction_events (
	id VARCHAR(64) NOT NULL,
	model_id VARCHAR(64) NOT NULL,
	occurred_at VARCHAR(32) NOT NULL,
	horizon_end VARCHAR(32) NOT NULL,
	subject VARCHAR(64) NOT NULL,
	source VARCHAR(24) NOT NULL,
	scenario VARCHAR(60) NOT NULL,
	features JSON NOT NULL,
	probability FLOAT,
	latency_ms FLOAT NOT NULL,
	valid INTEGER NOT NULL,
	error_type VARCHAR(60),
	segment VARCHAR(30),
	segment_model_id VARCHAR(64),
	PRIMARY KEY (id),
	CONSTRAINT event_valid CHECK (valid IN (0, 1)),
	CONSTRAINT event_probability CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1)),
	FOREIGN KEY(model_id) REFERENCES bank_monitoring.models (id)
)

;

CREATE INDEX IF NOT EXISTS events_scope_time ON bank_monitoring.prediction_events (model_id, source, scenario, occurred_at);


CREATE TABLE IF NOT EXISTS bank_monitoring.runs (
	id VARCHAR(64) NOT NULL,
	model_id VARCHAR(64) NOT NULL,
	source VARCHAR(24) NOT NULL,
	scenario VARCHAR(60) NOT NULL,
	window_start VARCHAR(32) NOT NULL,
	window_end VARCHAR(32) NOT NULL,
	config_sha256 VARCHAR(64) NOT NULL,
	computed_at VARCHAR(32) NOT NULL,
	report JSON NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT run_scope UNIQUE (model_id, source, scenario, window_start, window_end, config_sha256),
	FOREIGN KEY(model_id) REFERENCES bank_monitoring.models (id)
)

;

CREATE INDEX IF NOT EXISTS runs_scope_end ON bank_monitoring.runs (model_id, source, scenario, window_end);


CREATE TABLE IF NOT EXISTS bank_monitoring.alerts (
	id VARCHAR(64) NOT NULL,
	run_id VARCHAR(64) NOT NULL,
	rule VARCHAR(40) NOT NULL,
	active INTEGER NOT NULL,
	first_seen VARCHAR(32) NOT NULL,
	last_seen VARCHAR(32) NOT NULL,
	evidence JSON NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(run_id) REFERENCES bank_monitoring.runs (id)
)

;

CREATE INDEX IF NOT EXISTS alerts_run ON bank_monitoring.alerts (run_id);


CREATE TABLE IF NOT EXISTS bank_monitoring.outcomes (
	event_id VARCHAR(64) NOT NULL,
	label INTEGER NOT NULL,
	observed_at VARCHAR(32) NOT NULL,
	horizon_end VARCHAR(32) NOT NULL,
	source VARCHAR(30) NOT NULL,
	PRIMARY KEY (event_id),
	CONSTRAINT outcome_binary CHECK (label IN (0, 1)),
	FOREIGN KEY(event_id) REFERENCES bank_monitoring.prediction_events (id) ON DELETE CASCADE
)

;

REVOKE ALL ON SCHEMA bank_monitoring FROM PUBLIC;

DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'bank_monitoring_writer') THEN CREATE ROLE bank_monitoring_writer NOLOGIN; END IF; END $$;

DO $$ BEGIN EXECUTE format('GRANT bank_monitoring_writer TO %I', current_user); END $$;

GRANT USAGE ON SCHEMA bank_monitoring TO bank_monitoring_writer;

ALTER TABLE bank_monitoring."models" ENABLE ROW LEVEL SECURITY;

GRANT SELECT, INSERT ON bank_monitoring."models" TO bank_monitoring_writer;

DROP POLICY IF EXISTS monitoring_service ON bank_monitoring."models";

CREATE POLICY monitoring_service ON bank_monitoring."models" TO bank_monitoring_writer USING (true) WITH CHECK (true);

ALTER TABLE bank_monitoring."schema_version" ENABLE ROW LEVEL SECURITY;

GRANT SELECT ON bank_monitoring."schema_version" TO bank_monitoring_writer;

DROP POLICY IF EXISTS monitoring_service ON bank_monitoring."schema_version";

CREATE POLICY monitoring_service ON bank_monitoring."schema_version" TO bank_monitoring_writer USING (true) WITH CHECK (true);

ALTER TABLE bank_monitoring."prediction_events" ENABLE ROW LEVEL SECURITY;

GRANT SELECT, INSERT, UPDATE, DELETE ON bank_monitoring."prediction_events" TO bank_monitoring_writer;

DROP POLICY IF EXISTS monitoring_service ON bank_monitoring."prediction_events";

CREATE POLICY monitoring_service ON bank_monitoring."prediction_events" TO bank_monitoring_writer USING (true) WITH CHECK (true);

ALTER TABLE bank_monitoring."runs" ENABLE ROW LEVEL SECURITY;

GRANT SELECT, INSERT, UPDATE ON bank_monitoring."runs" TO bank_monitoring_writer;

DROP POLICY IF EXISTS monitoring_service ON bank_monitoring."runs";

CREATE POLICY monitoring_service ON bank_monitoring."runs" TO bank_monitoring_writer USING (true) WITH CHECK (true);

ALTER TABLE bank_monitoring."alerts" ENABLE ROW LEVEL SECURITY;

GRANT SELECT, INSERT, UPDATE ON bank_monitoring."alerts" TO bank_monitoring_writer;

DROP POLICY IF EXISTS monitoring_service ON bank_monitoring."alerts";

CREATE POLICY monitoring_service ON bank_monitoring."alerts" TO bank_monitoring_writer USING (true) WITH CHECK (true);

ALTER TABLE bank_monitoring."outcomes" ENABLE ROW LEVEL SECURITY;

GRANT SELECT, INSERT ON bank_monitoring."outcomes" TO bank_monitoring_writer;

DROP POLICY IF EXISTS monitoring_service ON bank_monitoring."outcomes";

CREATE POLICY monitoring_service ON bank_monitoring."outcomes" TO bank_monitoring_writer USING (true) WITH CHECK (true);

INSERT INTO bank_monitoring.schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
