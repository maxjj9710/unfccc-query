-- DROP DATABASE ENERGY
USE MASTER
CREATE DATABASE ENERGY
GO 

USE ENERGY
GO

CREATE TABLE CATEGORY (
   CategoryID INT PRIMARY KEY,
   CategoryName varchar(255) NOT NULL,
   Tag VARCHAR(255) NULL,
   CategoryDescr VARCHAR(255) NULL,
   DepthLvl INT NULL
)
GO
 

CREATE TABLE [YEAR] (
   YearID INT PRIMARY KEY,
   [YEAR] VARCHAR(255) NOT NULL
)
GO

CREATE TABLE [CLASSIFICATION] (
   ClassID INT PRIMARY KEY,
   ClassName varchar(255) NOT NULL,
   ClassDescr varchar(255) NULL
)
GO
 
CREATE TABLE COMBINATION (
   ComboID INT IDENTITY(1, 1) PRIMARY KEY,
   CategoryID INT NOT NULL,
   ClassID INT NOT NULL,
   YearID INT NOT NULL
)
GO
 

CREATE TABLE PARTY (
    PartyID INT PRIMARY KEY,
    AnnexID INT NOT NULL,
    PartyName VARCHAR(60) NOT NULL,
	PartyCode VARCHAR(30) NOT NULL
)
GO
 
CREATE TABLE ANNEX (
    AnnexID INT PRIMARY KEY,
    AnnexName VARCHAR(30) NOT NULL,
    AnnexDescr VARCHAR(255) NULL
)
GO
 
CREATE TABLE UNIT (
    UnitID INT PRIMARY KEY,
    UnitName VARCHAR(60) NOT NULL,
    UnitDescr VARCHAR(255) NULL
)
GO
 
CREATE TABLE GAS (
    GasID INT PRIMARY KEY,
    GasName VARCHAR(60) NOT NULL,
    GasDescr VARCHAR(255) NULL
)
GO
 
CREATE TABLE QUERY (
    QueryID INT IDENTITY(1, 1) PRIMARY KEY,
    ComboID INT NOT NULL,
    PartyID INT NOT NULL,
    MeasureID INT NOT NULL,
	GasID INT NOT NULL,
	UnitID INT NOT NULL,
	NumberValue FLOAT NULL,
	StringValue VARCHAR(255) NULL
)
GO
 
 
CREATE TABLE MEASUREMENT (
	MeasureID INT PRIMARY KEY,
	MeasureTypeID INT NOT NULL,
	MeasureName VARCHAR(255) NOT NULL,
	MeasureDescr VARCHAR(255) NULL
)
GO

CREATE TABLE MEASUREMENTTYPE(
	MeasureTypeID INT PRIMARY KEY,
	MeasureTypeName VARCHAR(255) NOT NULL,
	MeasureTypeDescr VARCHAR(255) NULL
)
GO

ALTER TABLE PARTY 
ADD CONSTRAINT FK_AnnexID
FOREIGN KEY (AnnexID)
REFERENCES ANNEX(AnnexID)
GO

ALTER TABLE QUERY 
ADD CONSTRAINT FK_Combo
FOREIGN KEY (ComboID)
REFERENCES COMBINATION(ComboID)
GO 

ALTER TABLE QUERY 
ADD CONSTRAINT FK_P_ID
FOREIGN KEY (PartyID)
REFERENCES PARTY(PartyID)
GO 

ALTER TABLE QUERY 
ADD CONSTRAINT FK_MID
FOREIGN KEY (MeasureID)
REFERENCES MEASUREMENT(MeasureID)
GO 

ALTER TABLE QUERY 
ADD CONSTRAINT FK_gas
FOREIGN KEY (GasID)
REFERENCES GAS(GasID)
GO 

ALTER TABLE QUERY 
ADD CONSTRAINT FK_unit
FOREIGN KEY (UnitID)
REFERENCES UNIT(UnitID)
GO 
 
ALTER TABLE MEASUREMENT
ADD CONSTRAINT FK_MTID
FOREIGN KEY (MeasureTypeID)
REFERENCES MEASUREMENTTYPE(MeasureTypeID)
GO 

ALTER TABLE COMBINATION
ADD CONSTRAINT FK_CategoryID
FOREIGN KEY (CategoryID)
REFERENCES CATEGORY(CategoryID)
GO
 
ALTER TABLE COMBINATION
ADD CONSTRAINT FK_ClassID
FOREIGN KEY (ClassID)
REFERENCES [CLASSIFICATION](ClassID)
GO
 
ALTER TABLE COMBINATION
ADD CONSTRAINT FK_YearID
FOREIGN KEY (YearID)
REFERENCES [YEAR](YearID)
GO


CREATE PROCEDURE Get_PartyID
@Party_code VARCHAR(20),
@Party_ID INT OUTPUT 
AS
SET @Party_ID = (SELECT PartyID FROM PARTY WHERE PartyCode = @Party_code)
GO

CREATE PROCEDURE Get_Category_ID 
@Cat_Name VARCHAR(255),
@Cat_ID INT OUTPUT
AS
SET @Cat_ID = (SELECT CategoryID FROM CATEGORY WHERE CategoryName = @Cat_Name)
GO

CREATE PROCEDURE Get_Class_ID
@Class_Name VARCHAR(255),
@Class_ID INT OUTPUT
AS
SET @Class_ID = (SELECT ClassID FROM [CLASSIFICATION] WHERE ClassName = @Class_Name)
GO

CREATE PROCEDURE Get_Measure_ID
@M_Name VARCHAR(255),
@M_ID INT OUTPUT
AS
SET @M_ID = (SELECT MeasureID FROM MEASUREMENT WHERE MeasureName = @M_Name)
GO

CREATE PROCEDURE Get_Gas_ID
@Gas_Name VARCHAR(255),
@Gas_ID INT OUTPUT
AS
SET @Gas_ID = (SELECT GasID FROM GAS WHERE GasName = @Gas_Name)
GO

CREATE PROCEDURE Get_Unit_ID
@Unit_Name VARCHAR(255),
@Unit_ID INT OUTPUT
AS
SET @Unit_ID = (SELECT UnitID FROM UNIT WHERE UnitName = @Unit_Name)
GO

CREATE PROCEDURE Get_Year_ID 
@Year_Name VARCHAR(255),
@Year_ID INT OUTPUT
AS
SET @Year_ID = (SELECT YearID FROM [YEAR] WHERE [YEAR] = @Year_Name)
GO

CREATE PROCEDURE Get_Combo_ID 
@Cl_Name VARCHAR(255),
@Categ_Name VARCHAR(255),
@Year_N VARCHAR(10),
@Cb_ID INT OUTPUT
AS
SET @Cb_ID = (SELECT C.ComboID FROM COMBINATION C
				JOIN [CLASSIFICATION] CLS ON C.ClassID = CLS.ClassID
				JOIN [YEAR] Y ON C.YearID = Y.YearID
				JOIN CATEGORY CG ON C.CategoryID = CG.CategoryID
				WHERE Y.[YEAR] = @Year_N
				AND CLS.ClassName = @Cl_Name
				AND CG.CategoryName = @Categ_Name)
GO

CREATE PROCEDURE Insert_Query_Combo
@Party_cody VARCHAR(255),
@Cat_Namy VARCHAR(255),
@Class_Namy VARCHAR (255),
@M_Namy VARCHAR(255),
@Gas_Namy VARCHAR(255),
@Unit_Namy VARCHAR(255),
@Year_Namy VARCHAR(255),
@NumVal FLOAT,
@StrVal VARCHAR(255)
AS
DECLARE @Combo_ID INT, @Party_ID INT, @Measure_ID INT, @Gas_ID INT, @Unit_ID INT, @Class_ID INT,
@Year_ID INT, @Cate_ID INT

EXEC Get_Class_ID
@Class_Name = @Class_Namy,
@Class_ID = @Class_ID OUTPUT

EXEC Get_Year_ID 
@Year_Name = @Year_Namy,
@Year_ID = @Year_ID OUTPUT

EXEC Get_Category_ID 
@Cat_Name = @Cat_Namy,
@Cat_ID = @Cate_ID OUTPUT

EXEC Get_PartyID
@Party_code = @Party_cody,
@Party_ID = @Party_ID OUTPUT 

EXEC Get_Measure_ID
@M_Name = @M_Namy,
@M_ID = @M_Namy OUTPUT

EXEC Get_Gas_ID
@Gas_Name = @Gas_Namy,
@Gas_ID = @Gas_ID OUTPUT

EXEC Get_Unit_ID
@Unit_Name = @Unit_Namy,
@Unit_ID = @Unit_ID OUTPUT

IF NOT EXISTS (SELECT * FROM COMBINATION 
			WHERE CategoryID = @Cate_ID
			AND ClassID = @Class_ID
			AND YearID = @Year_ID)
BEGIN 
	BEGIN TRANSACTION T1
	INSERT INTO COMBINATION (CategoryID, YearID, ClassID)
	VALUES (@Cate_ID, @Year_ID, @Class_ID)
	SET @Combo_ID = (SELECT SCOPE_IDENTITY())
	INSERT INTO QUERY (ComboID, PartyID, MeasureID, GasID, UnitID, NumberValue, StringValue)
	VALUES (@Combo_ID, @Party_ID, @Measure_ID, @Gas_ID, @Unit_ID, @NumVal, @StrVal)
	COMMIT TRANSACTION T1
END

ELSE
BEGIN
	BEGIN TRANSACTION T1
	SET @Combo_ID = (SELECT ComboID FROM COMBINATION
	WHERE YearID = @Year_ID AND ClassID = @Class_ID AND CategoryID = @Cate_ID)
	INSERT INTO QUERY (ComboID, PartyID, MeasureID, GasID, UnitID, NumberValue, StringValue)
	VALUES (@Combo_ID, @Party_ID, @Measure_ID, @Gas_ID, @Unit_ID,  @NumVal, @StrVal) 
	COMMIT TRANSACTION T1
END

GO


