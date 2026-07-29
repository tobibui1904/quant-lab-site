import duckdb

# create a connection to a file called 'file.db'
con = duckdb.connect("quant_trading.db")
# create a table and load data into it
# con.sql("CREATE TABLE Assets (id INTEGER PRIMARY KEY, sector VARCHAR NOT NULL, Asset1 VARCHAR NOT NULL, Asset2 VARCHAR NOT NULL)")
# con.sql("CREATE TABLE Portfolio (id INTEGER)")
# con.sql("CREATE TABLE Trading_Log (Date date NOT NULL, Asset VARCHAR NOT NULL, Action VARCHAR NOT NULL ,Price FLOAT NOT NULL, Quantity INTEGER NOT NULL, Trade_Type VARCHAR NOT NULL)")

# query the table
con.table("Trading_Log").show()

# Reset portfolio database
# con.execute("DELETE FROM Trading_Log")