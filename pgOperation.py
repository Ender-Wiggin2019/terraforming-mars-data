#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine


class PgOperation():

    def __init__(self, ip, port, user, pwd, db, schema='public'):
        self.ip = ip
        self.port = port
        self.user = user
        self.pwd = pwd
        self.db = db
        self.schema = schema

    def readTable(self, table): # usage: read table from public
        pg_local = [self.ip, self.port, self.user, self.pwd , self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        try:
            cur = conn.cursor()
            # Use double quotes to handle special characters in table names
            sql = 'SELECT * FROM "%s"."%s"'%(self.schema,table)
            df = pd.read_sql(sql,con=conn)
            cur.close()
            return df
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def readSql(self, sql): # usage: read sql from public
        pg_local = [self.ip, self.port, self.user, self.pwd, self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        try:
            cur = conn.cursor()
            df = pd.read_sql(sql,con=conn)
            cur.close()
            return df
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def writeExcelToPg(self,filename,target_table_name,f_sheet_name=0,f_skiprows=[]):  # usage: write a excel file to pgsql public scehma
        mapping = pd.read_excel(filename,sheet_name=f_sheet_name,skiprows=f_skiprows)
        engine = create_engine('postgresql+psycopg2://%s:%s@%s:%s/%s'%(self.user, self.pwd, self.ip, self.port, self.db))
        connection = engine.connect()
        # pg_df.head(0).to_sql('table_name', connection, if_exists='replace',index=False) #drops old table and creates new empty table

        conn = engine.raw_connection()
        cur = conn.cursor()
        mapping.to_sql(target_table_name, connection, if_exists='replace',index=False,schema=self.schema)
        # cur.execute('ALTER TABLE public.mapping OWNER to bi')
        engine.dispose()

    def writeDfToPg(self,df,target_table_name):  # usage: write a excel file to pgsql public scehma
        return None
        engine = create_engine('postgresql+psycopg2://%s:%s@%s:%s/%s'%(self.user, self.pwd, self.ip, self.port, self.db))
        connection = engine.connect()
        # pg_df.head(0).to_sql('table_name', connection, if_exists='replace',index=False) #drops old table and creates new empty table

        conn = engine.raw_connection()
        cur = conn.cursor()
        df.to_sql(target_table_name, connection, if_exists='replace',index=False,schema=self.schema)
        # cur.execute('ALTER TABLE public.mapping OWNER to bi')
        engine.dispose()

    def appendPgTable(self,df,table): # append df to a pgsql table
        return None
        # pg_df["invoice_money_done"] = pg_df["invoice_money_done"].astype(float)
        engine = create_engine('postgresql+psycopg2://%s:%s@%s:%s/%s'%(self.user, self.pwd, self.ip, self.port, self.db))
        connection = engine.connect()
        # df.head(0).to_sql(table, connection, if_exists='replace',index=False) #drops old table and creates new empty table
        df.to_sql(table, connection, if_exists='append',index=False,schema=self.schema)
        connection.close()
        engine.dispose()

    def runProcedure(self,procedure):
        return None
        pg_local = [self.ip, self.port, self.user, self.pwd , self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        try:
            cur = conn.cursor()
            sql = 'CALL "%s"."%s";'%(self.schema,procedure)
            cur.execute(sql)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def run(self,sql):
        pg_local = [self.ip, self.port, self.user, self.pwd , self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def deleteTableData(self, table):
        return None
        pg_local = [self.ip, self.port, self.user, self.pwd , self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        sql = 'TRUNCATE "%s"."%s"'%(self.schema,table)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        return 0

    def deleteRowsbyCondition(self,table,condition):
        return None
        pg_local = [self.ip, self.port, self.user, self.pwd , self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        sql = 'DELETE FROM "%s"."%s" WHERE %s'%(self.schema,table,condition)
        rows_deleted = 0
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows_deleted  = cur.rowcount
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        return rows_deleted

    def deleteAtivityRecordByDate(self,table,date):
        return None
        pg_local = [self.ip, self.port, self.user, self.pwd , self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        sql = 'DELETE FROM "%s"."%s" WHERE activity_date = %%s'%(self.schema,table)
        rows_deleted = 0
        try:
            cur = conn.cursor()
            cur.execute(sql, (date,))
            rows_deleted  = cur.rowcount
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        return rows_deleted

    def deleteRecordByDate(self,table, date):
        return None
        pg_local = [self.ip, self.port, self.user, self.pwd , self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        sql = 'DELETE FROM "%s"."%s" WHERE record_date = %%s'%(self.schema,table)
        rows_deleted = 0
        try:
            cur = conn.cursor()
            cur.execute(sql, (date,))
            rows_deleted  = cur.rowcount
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        return rows_deleted

    def writeHistoryExcelToPg(self,filename,target_table_name,f_sheet_name=0,f_skiprows=[]):  # usage: write a excel file to pgsql public scehma
        mapping = pd.read_excel(filename,sheet_name=f_sheet_name,skiprows=f_skiprows)
        engine = create_engine('postgresql+psycopg2://%s:%s@%s:%s/%s'%(self.user, self.pwd, self.ip, self.port, self.db))
        connection = engine.connect()
        # pg_df.head(0).to_sql('table_name', connection, if_exists='replace',index=False) #drops old table and creates new empty table

        conn = engine.raw_connection()
        cur = conn.cursor()
        mapping.to_sql(target_table_name, connection, if_exists='replace',index=False,schema=self.schema)
        # cur.execute('ALTER TABLE public.mapping OWNER to bi')
        engine.dispose()

    def listTables(self):  # usage: get all table names in the schema
        pg_local = [self.ip, self.port, self.user, self.pwd, self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        try:
            cur = conn.cursor()
            sql = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            cur.execute(sql, (self.schema,))
            tables = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            return tables
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            return []

    def tableToCsv(self, table, limit=10, outputPath=None):  # usage: export table data to CSV
        """
        Export first N rows of a table to CSV file.

        Args:
            table: table name
            limit: number of rows to export (default 10)
            outputPath: output CSV file path (default: {table}.csv)

        Returns:
            DataFrame of exported data
        """
        pg_local = [self.ip, self.port, self.user, self.pwd, self.db]
        conn = psycopg2.connect(host=pg_local[0], port=pg_local[1], user=pg_local[2], password=pg_local[3], database=pg_local[4])
        try:
            # Use double quotes to handle special characters in table names
            sql = 'SELECT * FROM "%s"."%s" LIMIT %d' % (self.schema, table, limit)
            df = pd.read_sql(sql, con=conn)
            conn.close()

            if outputPath is None:
                outputPath = f'{table}.csv'

            df.to_csv(outputPath, index=False)
            print(f'Exported {len(df)} rows from {table} to {outputPath}')
            return df
        except (Exception, psycopg2.DatabaseError) as error:
            print(f'Error exporting {table}: {error}')
            return None

    def exportAllTablesToSnapshot(self, snapshotDir='snapshot', limit=10):  # usage: export all tables to snapshot folder
        """
        Export first N rows of all tables to CSV files in snapshot directory.

        Args:
            snapshotDir: output directory for CSV files (default: 'snapshot')
            limit: number of rows to export per table (default 10)

        Returns:
            dict of {table_name: DataFrame}
        """
        import os

        # Create snapshot directory if not exists
        if not os.path.exists(snapshotDir):
            os.makedirs(snapshotDir)

        tables = self.listTables()
        print(f'Found {len(tables)} tables in schema {self.schema}')

        results = {}
        for table in tables:
            outputPath = os.path.join(snapshotDir, f'{table}.csv')
            df = self.tableToCsv(table, limit, outputPath)
            if df is not None:
                results[table] = df

        print(f'\nExported {len(results)} tables to {snapshotDir}/')
        return results
